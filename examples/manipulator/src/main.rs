#[tokio::main]
async fn main() {
    // initialize logger
    simple_logger::SimpleLogger::new().init().expect("unable to initialize logger");
    manipulator::start().await;
}

mod manipulator {
    use std::process::exit;
    use seigyo::{Matrix, KinematicsChain, Screw, Transformation};
    use time::Clock;
    use config::Config;
    use gz::{msgs::double::Double, transport::{Node, Publisher}};
    use tokio::{sync::OnceCell, time::Duration};

    static CONFIG: OnceCell<Config> = OnceCell::const_new();

    pub(crate) async fn start() {
        // initialize config
        let config = CONFIG.get_or_try_init(|| async { Config::load().await }).await.unwrap_or_else(|e| {
            log::error!("unable to load config: {e}");
            exit(1);
        });

        // create gz node
        let mut node = Node::new().unwrap_or_else(|| {
            log::error!("unable to create gz node");
            exit(2);
        });

        // initialize clock
        let clock = if config.sync_sim_clock {
            time::Clock::connect_sim(&mut node).await.unwrap_or_else(|e| {
                log::error!("unable to initialize clock. {e}");
                exit(3);
            })
        } else {
            time::Clock::connect_real()
        };

        // create manipulator
        let mut manipulator = Manipulator::<4>::init([0.; 4], &mut node, &clock).await.unwrap_or_else(|e| {
            log::error!("unable to initialize manipulator: {e}");
            exit(4);
        });

        // joint angles
        let theta_list = [
            0.,
            0.,
            core::f64::consts::FRAC_PI_2,
            core::f64::consts::FRAC_PI_2,
        ];

        log_fk_space(&manipulator, &theta_list);

        manipulator.set_joint_angles(&theta_list).await.unwrap_or_else(|e| {
            log::error!("unable to run manipulator: {e}");
            exit(5);
        });
    }


    fn log_fk_space(manipulator: &Manipulator<4>, theta_list: &[f64; 4]) {
        let rotation = Matrix::from([
            [0., 0., -1.],
            [1., 0., 0.],
            [0., -1., 0.],
        ]);
        
        let translation = Matrix::from([
           [- 0.15 - 0.295 - 0.221],
           [0.02 - 0.048],
           [0.244 + 0.045],
        ]);

        let m = Transformation::new(rotation, translation).unwrap_or_else(|e| {
            log::error!("unable to create transformation matrix: {e}");
            exit(6);
        });

        log::debug!("m:\n{m}");

        let s_list = [
            // omega = [0, 0, 1]
            // q = [0, 0, 0.084]
            Screw::new_revolute(Matrix::from([[0.], [0.], [1.]]), Matrix::from([[0.], [0.], [0.084]])).expect("invalid screw marix"),
            // omega = [0, 1, 0]
            // q = [-0.15, 0.02, 0.084 + 0.16] = [-0.15, 0.02, 0.244]
            Screw::new_revolute(Matrix::from([[0.], [1.], [0.]]), Matrix::from([[-0.15], [0.02], [0.084 + 0.16]])).expect("invalid screw marix"),
            // omega = [0, 1, 0]
            // q = [-0.15-0.295, 0.02, 0.084 + 0.16] = [-0.445, 0.02, 0.244]
            Screw::new_revolute(Matrix::from([[0.], [1.], [0.]]), Matrix::from([[-0.15-0.295], [0.02], [0.084 + 0.16]])).expect("invalid screw marix"),
            // omega = [0, 1, 1]
            // q = [-0.15-]
            Screw::new_revolute(Matrix::from([[-1.], [0.], [0.]]), Matrix::from([[- 0.15- 0.295 - 0.221], [0.02 - 0.048], [0.084 + 0.16 + 0.045]])).expect("invalid screw marix"),
        ];

        let t = manipulator.fk_space(m, s_list, *theta_list);
        log::info!("fk:\n{t}\n");
    }

    struct Manipulator<'cl, const N: usize> {
        state: [f64; N],
        joints: [Publisher<Double>; N],
        clock: &'cl Clock,
    }

    impl<'cl, const N: usize> Manipulator<'cl, N> {
        /// ## Initialize the manipulator
        /// Subscribes to joint topics. The prefix for the 
        /// 
        /// **Note:** This method blocks the thread for 100ms, waiting for the zmq bus to initialize. If this is not done, the first immediate message is skipped.
        async fn init(initial_state: [f64; N],node: &mut Node, clock: &'cl Clock) -> Result<Self, Error> {
            /// The prefix for joint topic
            const JOINT_TOPIC_PREFIX: &str = "/manipulator/joint";

            let maybe_joints = std::array::from_fn(|i| {
                let topic = format!("{JOINT_TOPIC_PREFIX}{i}");
                node.advertise(&topic)
            });

            if let Some(uninitialized_index) = maybe_joints.iter().position(Option::is_none) {
                return Err(Error::Init(uninitialized_index))
            }

            // wait for the connection using real (system) time,
            // regardless of which clock mode is active
            tokio::time::sleep(Duration::from_millis(100)).await;

            // wait for the simulation to start
            clock.wait_active().await;

            Ok(Self { state: initial_state, joints: maybe_joints.map(Option::unwrap), clock })
        }

        /// ## Move joints
        /// Moves the joints relative to the current position
        /// 
        /// **Note:** This method blocks the thread until all the joints are placed in the desired position
        async fn set_joint_angles(&mut self, theta_array: &[f64; N]) -> Result<(), Error> {
            log::debug!("setting joint angles: {theta_array:?}");
            const TIME_MILLIS: u64 = 1000;

            let vel_array = theta_array.map(|v| v * 1000. / TIME_MILLIS as f64);
            // start
            self.set_joint_speeds(&vel_array)?;

            // sleep
            self.clock.sleep(Duration::from_millis(TIME_MILLIS)).await;

            // stop
            self.set_joint_speeds(&[0.; N])?;

            // update joint angles
            self.state = *theta_array;

            Ok(())
        }

        /// ## Set joint velocity
        /// Sets the provided velocity to the joints
        fn set_joint_speeds(&mut self, vel_array: &[f64; N]) -> Result<(), Error> {
            log::debug!("setting joint velocities: {vel_array:?}");

            for (i, &vel) in vel_array.iter().enumerate() {
                // message
                let msg = Double {
                    data: vel,
                    ..Default::default()
                };

                // publish
                if ! self.joints[i].publish(&msg) {
                    return Err(Error::Publish(i));
                }
            }

            Ok(())
        }
    }

    impl<const N: usize> seigyo::KinematicsChain<N> for Manipulator<'_, N> {}

    mod time {
        use std::ops::Add;
        use gz::{msgs::clock::Clock as GzClock, transport::Node};
        use tokio::{sync::watch, task::spawn_blocking, time::{Duration, sleep, timeout}};

        use super::Error;

        /// The amount of time to wait for a clock message from the simulation
        const SIM_WAIT_TIMEOUT: u64 = 2000;

        #[derive(Debug, Default)]
        pub(super) enum Clock {
            #[default]
            /// Represents the system clock
            SystemClock,
            /// Represents the simulation clock
            SimClock(watch::Receiver<Time>),
        }

        impl Clock {
            /// Initializes the simulation clock.
            pub(crate) async fn connect_sim(node: &mut Node) -> Result<Self, Error> {
                let (tx, rx) = watch::channel(Time::default());

                const CLOCK_TOPIC: &str = "/clock";
                let receiver = node.subscribe_channel::<GzClock>(CLOCK_TOPIC, 10).ok_or(Error::Subscribe(CLOCK_TOPIC.to_string()))?;

                // receiver
                spawn_blocking(move ||
                    receiver
                        .iter()
                        .filter_map(|clock_msg| clock_msg.sim.as_ref().map(|sim| Time::new(sim.sec, sim.nsec)))
                        .for_each(|time| if tx.send(time).is_err() { log::error!("unable to set sim time") })
                );

                Ok(Self::SimClock(rx))
            }

            pub(super) fn connect_real() -> Self {
                Self::SystemClock
            }
            
            pub(super) async fn sleep(&self, duration: Duration) {
                match self {
                    Self::SystemClock => sleep(duration).await,
                    Self::SimClock(rx) => {
                        let mut rx = rx.clone();
                        let target = *rx.borrow() + duration;

                        // sleep for 'duration' amount of time because sim time is always greater than real time
                        sleep(duration).await;

                        // initial time overflow check
                        if *rx.borrow() >= target {
                            return;
                        }

                        loop {
                            rx.changed().await.expect("recieve error for clock watch channel");
                            if *rx.borrow() >= target {
                                break;
                            }
                        }
                    }
                }
            }

            /// Waits until the clock becomes active, logging a warning every [`SIM_WAIT_TIMEOUT`] ms.
            ///
            /// For [`Clock::SystemClock`] this returns immediately.
            ///
            /// For [`Clock::SimClock`] it blocks until simulation is playing (sim time > 0).
            pub(super) async fn wait_active(&self) {
                if let Self::SimClock(rx) = self {
                    if rx.borrow().to_nanos() > 0 { return; }
                    let mut rx = rx.clone();

                    let duration = Duration::from_millis(SIM_WAIT_TIMEOUT);
                    loop {
                        match timeout(duration, rx.changed()).await {
                            Ok(Ok(_)) => {
                                if rx.borrow().to_nanos() > 0 { break; }
                                log::warn!("simulation not initiated yet. will check in {SIM_WAIT_TIMEOUT} ms");

                                sleep(duration).await;
                                // mark the latest value seen
                                rx.borrow_and_update();
                            }
                            Ok(Err(_)) => {
                                log::error!("clock handle dropped");
                                return;
                            },
                            Err(_) => {
                                log::warn!("gazebo instance not running. will check in {SIM_WAIT_TIMEOUT} ms");
                            },
                        }
                    }
                }
            }
        }

        #[derive(Debug, Default, Clone, Copy, PartialEq, PartialOrd)]
        pub(super) struct Time {
            sec: u64,
            nsec: u32,
        }

        impl Time {
            fn new(sec: i64, nsec: i32) -> Self {
                assert!(sec >= 0, "simulation seconds cannot be negative");
                assert!(nsec >= 0, "simulation nanoseconds cannot be negative");

                Self { sec: sec as u64, nsec: nsec as u32 }
            }
            
            #[inline]
            fn to_nanos(&self) -> u128 {
                self.sec as u128 * 1_000_000_000 + self.nsec as u128
            }

            fn from_nanos(nanos: u128) -> Self {
                let sec = (nanos / 1_000_000_000) as u64;
                let nsec = (nanos % 1_000_000_000) as u32;
                Self { sec, nsec }
            }
        }

        impl Add<Duration> for Time {
            type Output = Self;

            fn add(self, dur: Duration) -> Self::Output {
                let nsec = self.to_nanos() + dur.as_nanos();
                Self::from_nanos(nsec)
            }
        }
    }

    mod config {
        use std::env::args;
        use tokio::fs::read_to_string;
        use serde::Deserialize;

        #[derive(Debug, Deserialize)]
        pub(super) struct Config {
            pub(super) sync_sim_clock: bool,
        }

        impl Config {
            pub(super) async fn load() -> Result<Self, Error> {
                const DEFAULT_PATH: &str = "config.toml";

                let mut args = args();

                let path;
                let path = if let Some(c) = args.nth(1) && (c == "-c" || c == "--config") {
                    path = args.next().ok_or(Error::MissingPath)?;
                    path.as_str()
                } else {
                    DEFAULT_PATH
                };

                log::info!("loading config from {path}");
                let contents = read_to_string(path).await?;
                let config = toml::from_str(&contents)?;

                log::info!("loaded config: {config:?}");
                Ok(config)
            }
        }

        #[derive(Debug)]
        pub(super) enum Error {
            MissingPath,
            Io(std::io::Error),
            Parse(toml::de::Error),
        }

        impl std::fmt::Display for Error {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    Self::MissingPath => write!(f, "missing config path"),
                    Self::Io(e) => write!(f, "io error: {e}"),
                    Self::Parse(e) => write!(f, "parse error: {e}"),
                }
            }
        }

        impl std::error::Error for Error {}

        impl From<std::io::Error> for Error {
            fn from(e: std::io::Error) -> Self {
                Self::Io(e)
            }
        }

        impl From<toml::de::Error> for Error {
            fn from(e: toml::de::Error) -> Self {
                Self::Parse(e)
            }
        }
    }

    #[derive(Debug)]
    enum Error {
        Init(usize),
        Publish(usize),
        Subscribe(String),
        Config(config::Error),
    }

    impl core::fmt::Display for Error {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Init(joint) => write!(f, "initialization error for joint: {joint}"),
                Self::Publish(joint) => write!(f, "error publishing to topic for joint {joint}"),
                Self::Subscribe(topic) => write!(f, "unable to subscribe to topic {topic}"),
                Self::Config(e) => write!(f, "configuration error: {e}"),
            }
        }
    }

    impl std::error::Error for Error {} 

    impl From<config::Error> for Error {
        fn from(e: config::Error) -> Self {
            Self::Config(e)
        }
    }
}
