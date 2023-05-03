use std::fmt::{Display, Formatter};

#[derive(Debug)]
pub struct Point {
    pub x: f32,
    pub y: f32
}

impl Display for Point{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "({},{})", self.x, self.y)
    }
}

impl From<(f32, f32)> for Point {
    fn from(value: (f32, f32)) -> Self {
        Point{
            x: value.0, y:value.1
        }
    }
}

impl From<Point> for (f32, f32) {
    fn from(value: Point) -> Self {
        (value.x.clone(), value.y.clone())
    }
}

impl Into<Point> for Vec<f32> {
    fn into(self) -> Point {
        Point {
            x: self[0],
            y: self[1]
        }
    }
}

