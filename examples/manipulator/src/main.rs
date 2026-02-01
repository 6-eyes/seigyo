use crate::control::Manipulator;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut manipulator = Manipulator::init()?;
    manipulator.publish(2, 1.0)?;

    Ok(())
}

mod control {
    use gz::{msgs::double::Double, transport::{Node, Publisher}};
    use super::Error;

    #[derive(Debug)]
    pub struct Manipulator {
        joint_1: Publisher<Double>,
        joint_2: Publisher<Double>,
    }

    impl Manipulator {
        pub fn init() -> Result<Self, Error> {
            let mut node = Node::new().unwrap();
            let topic1 = "/manipulator/joint1";
            let topic2 = "/manipulator/joint2";

            let manipulator = Self {
                joint_1: node.advertise(topic1).ok_or(Error::Init("unable to create publisher"))?,
                joint_2: node.advertise(topic2).ok_or(Error::Init("unable to create publisher"))?,
            };

            // wait for the connection
            std::thread::sleep(std::time::Duration::from_millis(100));
            Ok(manipulator)
        }

        /// publish velocity in radians per second
        pub fn publish(&mut self, joint: u8, data: f64) -> Result<(), Error> {
            let mut data = Double {
                data,
                ..Default::default() 
            };

            let joint = match joint {
                1 => &mut self.joint_1,
                2 => &mut self.joint_2,
                j => panic!("invalid joint {j}"),
            };


            if ! joint.publish(&data) {
                return Err(Error::Publish(1));
            }

            std::thread::sleep(std::time::Duration::from_millis(1500));

            data.data = 0.;

            if ! joint.publish(&data) {
                return Err(Error::Publish(1));
            }
            
            Ok(())
        }
    }
}

#[derive(Debug)]
enum Error {
    Init(&'static str),
    Publish(u8),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Init(msg) => write!(f, "initialization error: {msg}"),
            Self::Publish(joint) => write!(f, "error publishing to topic for joint {joint}"),
        }
    }
}

impl std::error::Error for Error {}