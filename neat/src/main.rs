use foliage::Foliage;
const VIEW_AREA: (f32, f32) = (1800.0, 800.0);
fn main() {
    let mut foliage = Foliage::new();
    foliage.set_desktop_size(VIEW_AREA);
    foliage.photosynthesize();
}
