use crate::control::Manipulator;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut manipulator = Manipulator::<2>::init()?;
    manipulator.set_joint_angles(&[ 0., std::f64::consts::FRAC_PI_2 ])?;

    Ok(())
}

mod control {
    use gz::{msgs::double::Double, transport::{Node, Publisher}};
    use super::Error;

    const TOPIC_PREFIX: &str = "/manipulator/joint";

    #[derive(Debug)]
    pub struct Manipulator<const N: usize> {
        joints: [Publisher<Double>; N],
    }

    impl<const N: usize> Manipulator<N> {
        /// ## Initialize the manipulator
        /// Subscribes to joint topics. The prefix for the 
        /// 
        /// **Note:** This method blocks the thread for 100ms, waiting for the zmq bus to initialize. If this is not done, the first immediate message is skipped.
        pub fn init() -> Result<Self, Error> {
            let mut node = Node::new().unwrap();

            let maybe_joints = std::array::from_fn(|i| {
                let topic = format!("{TOPIC_PREFIX}{i}");
                node.advertise(&topic)
            });

            if let Some(uninitialized_index) = maybe_joints.iter().position(Option::is_none) {
                return Err(Error::Init(uninitialized_index))
            }

            // wait for the connection
            std::thread::sleep(std::time::Duration::from_millis(100));

            Ok(Self { joints: maybe_joints.map(Option::unwrap) })
        }

        /// ## Move joints
        /// Moves the joints relative to the current position
        /// 
        /// **Note:** This method blocks the thread until all the joints are placed in the desired position
        pub fn set_joint_angles(&mut self, theta_array: &[f64; N]) -> Result<(), Error> {
            log::debug!("setting jount angles: {theta_array:?}");
            const TIME_MILLIS: u64 = 1000;

            // start
            let vel_array = theta_array.map(|v| v * 1000. / TIME_MILLIS as f64);
            self.set_joint_speeds(&vel_array)?;

            std::thread::sleep(std::time::Duration::from_millis(TIME_MILLIS));
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
}

#[derive(Debug)]
enum Error {
    Init(usize),
    Publish(usize),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Init(joint) => write!(f, "initialization error for joint: {joint}"),
            Self::Publish(joint) => write!(f, "error publishing to topic for joint {joint}"),
        }
    }
}

impl std::error::Error for Error {}