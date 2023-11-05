extern crate tch;

use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Tensor, Kind};

const INPUT_SIZE: i64 = 6;
const HIDDEN_NODES: i64 = 128;
const OUTPUT_SIZE: i64 = 4;

#[derive(Debug)]
pub struct Net {
    fc1: nn::Linear,
    fc2: nn::Linear,
    fc3: nn::Linear,
}


impl Net {
    pub fn new(vs: &nn::Path) -> Net {
        let fc1 = nn::linear(vs, INPUT_SIZE, HIDDEN_NODES, Default::default());
        let fc2 = nn::linear(vs, HIDDEN_NODES, HIDDEN_NODES, Default::default());
        let fc3 = nn::linear(vs, HIDDEN_NODES, OUTPUT_SIZE, Default::default());
        Net { fc1, fc2, fc3 }
    }
}

impl Module for Net {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.fc1)
            .relu()
            .apply(&self.fc2)
            .relu()
            .apply(&self.fc3)
            .softmax(-1, Kind::Float)
    }
}

pub struct AI {
    pub net: Net,
    pub opt: nn::Optimizer<nn::Adam>,
}

impl AI {
    pub fn new(device: Device) -> AI {
        let vs = nn::VarStore::new(device);
        let net = Net::new(&vs.root());
        let opt = nn::Adam::default().build(&vs, 1e-3).unwrap();
        AI { net, opt }
    }


    pub fn train(&mut self, state: &Tensor, target: &Tensor) {
        let loss = self.net.forward(state).mse_loss(target, tch::Reduction::Mean);
        self.opt.backward_step(&loss);
    }


    pub fn choose_action(&self, state: &Tensor) -> i64 {
        self.net.forward(state).argmax(-1, false).int64_value(&[])
    }
}
