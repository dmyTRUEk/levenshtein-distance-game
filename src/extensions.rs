//! Some useful extensions

#![allow(dead_code)]

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

pub trait ToStrings {
	fn to_strings(self) -> (String, String);
}
impl ToStrings for (&str, &str) {
	fn to_strings(self) -> (String, String) {
		let (str1, str2) = self;
		(str1.to_string(), str2.to_string())
	}
}

