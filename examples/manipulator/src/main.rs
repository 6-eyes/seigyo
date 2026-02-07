
use simple_logger::SimpleLogger;

#[tokio::main]
async fn main() {
    // initialize logger
    SimpleLogger::new().init().expect("unable to initialize logger");
    control::start().await;
}

mod control {
    use std::process::exit;
    use gz::{msgs::double::Double, transport::{Node, Publisher}};
    use tokio::time::Duration;
    use time::SimClock;
    use super::Error;

    pub async fn start() {
        let mut node = Node::new().unwrap_or_else(|| {
            log::error!("unable to create gz node");
            exit(1);
        });

        let clock = time::SimClock::connect(&mut node).await.unwrap_or_else(|e| {
            log::error!("unable to initialize clock. {e}");
            exit(2);
        });

        let mut manipulator = Manipulator::<2>::init(&mut node, &clock).await.unwrap_or_else(|e| {
            log::error!("unable to initialize manipulator: {e}");
            exit(3);
        });

        manipulator.set_joint_angles(&[ 0., std::f64::consts::FRAC_PI_2 ]).await.unwrap_or_else(|e| {
            log::error!("unable to run manipulator: {e}");
            exit(4);
        });
    }

    pub struct Manipulator<'a, const N: usize> {
        joints: [Publisher<Double>; N],
        clock: &'a SimClock,
    }

    impl<'a, const N: usize> Manipulator<'a, N> {
        /// ## Initialize the manipulator
        /// Subscribes to joint topics. The prefix for the 
        /// 
        /// **Note:** This method blocks the thread for 100ms, waiting for the zmq bus to initialize. If this is not done, the first immediate message is skipped.
        pub async fn init(node: &mut Node, clock: &'a SimClock) -> Result<Self, Error> {
            /// The prefix for joint topic
            const JOINT_TOPIC_PREFIX: &str = "/manipulator/joint";

            let maybe_joints = std::array::from_fn(|i| {
                let topic = format!("{JOINT_TOPIC_PREFIX}{i}");
                node.advertise(&topic)
            });

            if let Some(uninitialized_index) = maybe_joints.iter().position(Option::is_none) {
                return Err(Error::Init(uninitialized_index))
            }

            // wait for the connection
            clock.sleep(Duration::from_millis(100)).await;

            Ok(Self { joints: maybe_joints.map(Option::unwrap), clock })
        }

        /// ## Move joints
        /// Moves the joints relative to the current position
        /// 
        /// **Note:** This method blocks the thread until all the joints are placed in the desired position
        pub async fn set_joint_angles(&mut self, theta_array: &[f64; N]) -> Result<(), Error> {
            log::debug!("setting jount angles: {theta_array:?}");
            const TIME_MILLIS: u64 = 1000;

            let vel_array = theta_array.map(|v| v * 1000. / TIME_MILLIS as f64);
            // start
            self.set_joint_speeds(&vel_array)?;

            // sleep
            self.clock.sleep(Duration::from_millis(TIME_MILLIS)).await;

            // stop
            self.set_joint_speeds(&[0.; N])?;

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

    pub mod time {
        use std::ops::Add;

        use gz::{msgs::clock::Clock, transport::Node};
        use tokio::{sync::watch, task::spawn_blocking, time::{Duration, sleep, timeout}};
        use super::Error;

        #[derive(Debug, Default, Clone, Copy, PartialEq)]
        struct SimTime {
            sec: u64,
            nsec: u32,
        }

        impl SimTime {
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

        impl Add<Duration> for SimTime {
            type Output = Self;

            fn add(self, dur: Duration) -> Self::Output {
                let nsec = self.to_nanos() + dur.as_nanos();
                Self::from_nanos(nsec)
            }
        }

        impl PartialOrd for SimTime {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                self.to_nanos().partial_cmp(&other.to_nanos())
            }
        }

        impl From<(i64, i32)> for SimTime {
            fn from((sec, nsec): (i64, i32)) -> Self {
                assert!(sec >= 0, "simulation seconds cannot be negative");
                assert!(nsec >= 0, "simulation nanoseconds cannot be negative");

                Self {
                    sec: sec as u64,
                    nsec: nsec as u32,
                }
            }
        }

        /// Represents the simulation clock
        pub struct SimClock {
            rx: watch::Receiver<SimTime>,
        }

        impl SimClock {
            /// Initializes the simulation clock.
            /// 
            /// Blocks until simulation starts
            pub async fn connect(node: &mut Node) -> Result<Self, Error> {
                let (tx, mut rx) = watch::channel(SimTime::default());

                const CLOCK_TOPIC: &str = "/clock";
                let receiver = node.subscribe_channel::<Clock>(CLOCK_TOPIC, 10).ok_or(Error::Subscribe(CLOCK_TOPIC.to_string()))?;

                // receiver
                spawn_blocking(move || for clock_msg in receiver {
                    // set time
                    if clock_msg.sim.is_some() {
                        let time = (clock_msg.sim.sec, clock_msg.sim.nsec);
                        if tx.send(time.into()).is_err() {
                            log::error!("unable to set sim time");
                        }
                    }
                });

                // wait for the simulation to start
                loop {
                    match timeout(Duration::from_secs(2), rx.changed()).await {
                        Ok(Ok(_)) => {
                            log::info!("simulation connected");
                            break;
                        },
                        Ok(Err(_)) => {
                            log::error!("handle shutdown");
                            break;
                        },
                        Err(_) => log::warn!("waiting for simulation to start (no message on clock topic)"),
                    }
                }

                Ok(Self { rx })
            }

            /// Gets the current simulation time
            fn now(&self) -> SimTime {
                *self.rx.borrow()
            }

            /// Sleeps for the provided amount of time duration
            pub async fn sleep(&self, duration: Duration) {
                let target = self.now() + duration;

                // sleep for 'duration' amount of time because sim time is always greater than real time
                sleep(duration).await;

                let mut rx = self.rx.clone();

                // initial time overflow check
                if *rx.borrow() >= target {
                    return;
                }

                loop {
                    rx.changed().await.expect("recieve error for clock watch channel");
                    if *rx.borrow() > target {
                        break;
                    }
                }
            }
        }

        impl Clone for SimClock {
            fn clone(&self) -> Self {
                Self {
                    rx: self.rx.clone(),
                }
            }
        }
    }
}

#[derive(Debug)]
enum Error {
    Init(usize),
    Publish(usize),
    Subscribe(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Init(joint) => write!(f, "initialization error for joint: {joint}"),
            Self::Publish(joint) => write!(f, "error publishing to topic for joint {joint}"),
            Self::Subscribe(topic) => write!(f, "unable to subscribe to topic {topic}"),
        }
    }
}

impl std::error::Error for Error {} 