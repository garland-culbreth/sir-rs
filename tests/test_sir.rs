use sirrs::sir::Model;

#[test]
fn sir_init_popf() {
    let mut model = Model::new();
    model.configure(10, 1.0, 0.01, 0.0, 0.02, 0.03, 0.04);
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
        model.s_popf[(0, 0)],
        1.0 - model.i_popf_init - model.r_popf_init,
        "Bad s_popf[(0, 0)] initialization value, expected {} got {}.",
        1.0 - model.i_popf_init - model.r_popf_init,
        model.s_popf[(0, 0)]
    );
    assert_eq!(
        model.i_popf[(0, 0)],
        model.i_popf_init,
        "Bad i_popf[(0, 0)] initialization value, expected {} got {}.",
        model.i_popf_init,
        model.i_popf[(0, 0)],
    );
    assert_eq!(
        model.r_popf[(0, 0)],
        model.r_popf_init,
        "Bad r_popf[(0, 0)] initialization value, expected {} got {}.",
        model.r_popf_init,
        model.r_popf[(0, 0)],
    );
    for t in 1..model.length {
        assert_eq!(
            model.s_popf[(t, 0)],
            0.0,
            "Bad s_popf[t>0] initialization value, expected 0.0 got {}.",
            model.s_popf[(t, 0)]
        );
        assert_eq!(
            model.i_popf[(t, 0)],
            0.0,
            "Bad i_popf[t>0] initialization value, expected 0.0 got {}.",
            model.i_popf[(t, 0)]
        );
        assert_eq!(
            model.r_popf[(t, 0)],
            0.0,
            "Bad r_popf[t>0] initialization value, expected 0.0 got {}.",
            model.r_popf[(t, 0)]
        );
    }
}

#[test]
fn sir_run_euler() {
    let mut model = Model::new();
    model.configure(10, 1.0, 0.01, 0.0, 0.02, 0.03, 0.04);
    model.init_popf();
    model.run_euler();
    for t in 1..model.length {
        let dsdt = (-model.incidence_rate * model.s_popf[(t - 1, 0)] * model.i_popf[(t - 1, 0)])
            + (model.recovery_rate * model.i_popf[(t - 1, 0)]);
        let didt = (model.incidence_rate * model.s_popf[(t - 1, 0)] * model.i_popf[(t - 1, 0)])
            - (model.removal_rate * model.i_popf[(t - 1, 0)])
            - (model.recovery_rate * model.i_popf[(t - 1, 0)]);
        let drdt = model.removal_rate * model.i_popf[(t - 1, 0)];
        assert!(
            (model.s_popf[(t, 0)] >= 0.0) & (model.s_popf[(t, 0)] <= 1.0),
            "s_popf[(t, 0)] not in [0, 1] at time {}, got {}",
            t,
            model.s_popf[(t, 0)]
        );
        assert!(
            (model.i_popf[(t, 0)] >= 0.0) & (model.i_popf[(t, 0)] <= 1.0),
            "i_popf[(t, 0)] not in [0, 1] at time {}, got {}",
            t,
            model.i_popf[(t, 0)]
        );
        assert!(
            (model.r_popf[(t, 0)] >= 0.0) & (model.r_popf[(t, 0)] <= 1.0),
            "r_popf[(t, 0)] not in [0, 1] at time {}, got {}",
            t,
            model.r_popf[(t, 0)]
        );
        assert_eq!(
            model.s_popf[(t, 0)],
            model.s_popf[(t - 1, 0)] + dsdt,
            "Bad s_popf[(t, 0)] at time {}, expected {} got {}",
            t,
            model.s_popf[(t - 1, 0)] + dsdt,
            model.s_popf[(t, 0)]
        );
        assert_eq!(
            model.i_popf[(t, 0)],
            model.i_popf[(t - 1, 0)] + didt,
            "Bad i_popf[(t, 0)] at time {}, expected {} got {}",
            t,
            model.i_popf[(t - 1, 0)] + didt,
            model.i_popf[(t, 0)]
        );
        assert_eq!(
            model.r_popf[(t, 0)],
            model.r_popf[(t - 1, 0)] + drdt,
            "Bad r_popf[(t, 0)] at time {}, expected {} got {}",
            t,
            model.r_popf[(t - 1, 0)] + drdt,
            model.r_popf[(t, 0)]
        );
    }
}
