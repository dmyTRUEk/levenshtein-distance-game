//! Some useful extensions

pub trait VecPushed<T> {
	fn pushed(self, el: T) -> Self;
	fn pushed_opt(self, el: T) -> Self;
}
impl<T> VecPushed<T> for Vec<T> {
	fn pushed(mut self, el: T) -> Self {
		self.push(el);
		self
	}
	fn pushed_opt(mut self, el: T) -> Self {
	    self.push(el);
		self.shrink_to_fit();
		self
	}
}

