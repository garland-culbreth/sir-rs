pub mod sirrs;

use ndarray::Array;
use sirrs::sir::Model;

fn main() {
    let mut model: Model = Model {
        length: 365,
        i_popf_init: 0.01,
        r_popf_init: 0.0,
        incidence_rate: 0.04,
        removal_rate: 0.01,
        recovery_rate: 0.01,
        s_popf: Array::default(1),
        i_popf: Array::default(1),
        r_popf: Array::default(1),
    };
    model.init_popf();
    model.run_fdm_o1();
}
