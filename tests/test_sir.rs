use sirrs::sir::Model;
use nalgebra::DVector;

#[test]
fn sir_init_popf() {
    let mut model: Model = Model {
        length: 10,
        i_popf_init: 0.01,
        r_popf_init: 0.0,
        incidence_rate: 0.02,
        removal_rate: 0.03,
        recovery_rate: 0.04,
        s_popf: DVector::default(),
        i_popf: DVector::default(),
        r_popf: DVector::default(),
    };
    model.init_popf();
    assert_eq!(
        model.s_popf.shape(),
        (model.length, 1),
        "Bad s_popf dimensions, expected {:?} got {:?}.",
        (model.length, 1),
        model.s_popf.shape(),
    );
    assert_eq!(
        model.i_popf.shape(),
        (model.length, 1),
        "Bad i_popf dimensions, expected {:?} got {:?}.",
        (model.length, 1),
        model.i_popf.shape(),
    );
    assert_eq!(
        model.r_popf.shape(),
        (model.length, 1),
        "Bad r_popf dimensions, expected {:?} got {:?}.",
        (model.length, 1),
        model.r_popf.shape(),
    );
    assert_eq!(
        model.s_popf[0],
        1.0 - model.i_popf_init - model.r_popf_init,
        "Bad s_popf[0] initialization value, expected {} got {}.",
        1.0 - model.i_popf_init - model.r_popf_init,
        model.s_popf[0]
    );
    assert_eq!(
        model.i_popf[0], model.i_popf_init,
        "Bad i_popf[0] initialization value, expected {} got {}.",
        model.i_popf_init, model.i_popf[0],
    );
    assert_eq!(
        model.r_popf[0], model.r_popf_init,
        "Bad r_popf[0] initialization value, expected {} got {}.",
        model.r_popf_init, model.r_popf[0],
    );
    for t in 1..model.length {
        assert_eq!(
            model.s_popf[t], 0.0,
            "Bad s_popf[t>0] initialization value, expected 0.0 got {}.",
            model.s_popf[t]
        );
        assert_eq!(
            model.i_popf[t], 0.0,
            "Bad i_popf[t>0] initialization value, expected 0.0 got {}.",
            model.i_popf[t]
        );
        assert_eq!(
            model.r_popf[t], 0.0,
            "Bad r_popf[t>0] initialization value, expected 0.0 got {}.",
            model.r_popf[t]
        );
    }
}

#[test]
fn sir_run_fdm_o1() {
    let mut model: Model = Model {
        length: 10,
        i_popf_init: 0.01,
        r_popf_init: 0.0,
        incidence_rate: 0.02,
        removal_rate: 0.03,
        recovery_rate: 0.04,
        s_popf: DVector::default(),
        i_popf: DVector::default(),
        r_popf: DVector::default(),
    };
    model.init_popf();
    model.run_fdm_o1();
    for t in 1..model.length {
        let dsdt = (-model.incidence_rate * model.s_popf[t - 1] * model.i_popf[t - 1])
            + (model.recovery_rate * model.i_popf[t - 1]);
        let didt = (model.incidence_rate * model.s_popf[t - 1] * model.i_popf[t - 1])
            - (model.removal_rate * model.i_popf[t - 1])
            - (model.recovery_rate * model.i_popf[t - 1]);
        let drdt = model.removal_rate * model.i_popf[t - 1];
        assert!(
            (model.s_popf[t] >= 0.0) & (model.s_popf[t] <= 1.0),
            "s_popf[t] not in [0, 1] at time {}, got {}",
            t,
            model.s_popf[t]
        );
        assert!(
            (model.i_popf[t] >= 0.0) & (model.i_popf[t] <= 1.0),
            "i_popf[t] not in [0, 1] at time {}, got {}",
            t,
            model.i_popf[t]
        );
        assert!(
            (model.r_popf[t] >= 0.0) & (model.r_popf[t] <= 1.0),
            "r_popf[t] not in [0, 1] at time {}, got {}",
            t,
            model.r_popf[t]
        );
        assert_eq!(
            model.s_popf[t],
            model.s_popf[t - 1] + dsdt,
            "Bad s_popf[t] at time {}, expected {} got {}",
            t,
            model.s_popf[t - 1] + dsdt,
            model.s_popf[t]
        );
        assert_eq!(
            model.i_popf[t],
            model.i_popf[t - 1] + didt,
            "Bad i_popf[t] at time {}, expected {} got {}",
            t,
            model.i_popf[t - 1] + didt,
            model.i_popf[t]
        );
        assert_eq!(
            model.r_popf[t],
            model.r_popf[t - 1] + drdt,
            "Bad r_popf[t] at time {}, expected {} got {}",
            t,
            model.r_popf[t - 1] + drdt,
            model.r_popf[t]
        );
    }
}