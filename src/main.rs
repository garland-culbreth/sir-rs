pub mod sirrs;

use nalgebra::DVector;
use sirrs::sir::Model;
// use sirrs::dismod::Model;

fn main() {
    let mut model: Model = Model {
        length: 365,
        i_popf_init: 0.01,
        r_popf_init: 0.0,
        incidence_rate: 0.04,
        removal_rate: 0.01,
        recovery_rate: 0.01,
        s_popf: DVector::default(),
        i_popf: DVector::default(),
        r_popf: DVector::default(),
    };
    model.init_popf();
    model.run_fdm_o1();
}
