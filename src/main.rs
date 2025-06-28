pub mod sirrs;

use nalgebra::DVector;
use sirrs::dismod::Model;

fn main() {
    let mut model: Model = Model {
        length: 365,
        c_init: 0.01,
        iota: 0.01,
        rho: 0.01,
        chi: 0.001,
        omega: 0.001,
        s: DVector::default(),
        c: DVector::default(),
        rc: DVector::default(),
        ro: DVector::default(),
    };
    model.init_popf();
    model.run_fdm_o1();
}
