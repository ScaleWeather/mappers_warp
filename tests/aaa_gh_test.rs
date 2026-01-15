use ndarray::s;

#[test]
fn gh_test(){
    mut_slice(f64::NAN);
    indexing(f64::NAN);
    
    mut_slice(f64::INFINITY);
    indexing(f64::INFINITY);

    panic!();
}

fn mut_slice(v: f64) {
    let mut arr = ndarray::Array2::<f64>::zeros((5,5));
    dbg!(&arr);
    dbg!(&arr[[0,0]].is_nan());
    dbg!(&arr[[0,0]].is_finite());
    
    arr.slice_mut(s![0..1, 0..1]).fill(v);
    dbg!(&arr);
    dbg!(&arr[[0,0]].is_nan());
    dbg!(&arr[[0,0]].is_finite());
}

fn indexing(v: f64) {
    let mut arr = ndarray::Array2::<f64>::zeros((5,5));
    dbg!(&arr);
    dbg!(&arr[[0,0]].is_nan());
    dbg!(&arr[[0,0]].is_finite());
    
    arr[[0,0]] = v;
    dbg!(&arr);
    dbg!(&arr[[0,0]].is_nan());
    dbg!(&arr[[0,0]].is_finite());
}