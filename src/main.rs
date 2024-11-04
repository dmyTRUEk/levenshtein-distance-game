//! Calc Levenshtein distance between words.

#![deny(
	unsafe_code,
	unused_results,
)]

#![feature(
	gen_blocks,
	coroutines,
	coroutine_trait,
	iter_from_coroutine,
	let_chains,
	stmt_expr_attributes,
)]

use std::{iter::from_coroutine, ops::Add};

use clap::{Parser, arg};

mod extensions;
mod macros;
mod utils_io;

use extensions::{VecPushed, ToStrings};
use utils_io::prompt;



#[derive(Parser, Debug)]
#[clap(
	about,
	author,
	version,
	help_template = "\
		{before-help}{name} {version}\n\
		{about}\n\
		Author: {author}\n\
		\n\
		{usage-heading} {usage}\n\
		\n\
		{all-args}{after-help}\
	",
)]
struct CliArgsPre {
	/// Language: `eng` or `ukr`.
	#[arg(short, long, default_value="eng")]
	language: String,

	/// Word 1 , Word 2
	#[arg(short, long)]
	word12: Option<String>,
}

struct CliArgsPost {
	language: Language,
	word12: Option<[String; 2]>,
}
impl From<CliArgsPre> for CliArgsPost {
	fn from(CliArgsPre {
		language,
		word12,
	}: CliArgsPre) -> Self {
		Self {
			language: match language.as_str() {
				"eng" => Language::Eng,
				"ukr" => Language::Ukr,
				_ => panic!()
			},
			word12: word12
				.map(|word12| {
					word12.split_once(',').unwrap().to_strings().into()
				})
		}
	}
}



fn main() {
	let cli_args = CliArgsPre::parse();
	let cli_args = CliArgsPost::from(cli_args);

	match cli_args.language {
		Language::Eng => {
			main_with_lang::<{Language::ENG}>(cli_args);
		}
		Language::Ukr => {
			main_with_lang::<{Language::UKR}>(cli_args);
		}
	}
}

fn main_with_lang<const A: u8>(cli_args: CliArgsPost) {
	struct Localization {
		word: &'static str,
		input_word: &'static str,
		solution: &'static str,
		solution_len: &'static str,
		word_len: &'static str,
		points_float: &'static str,
		points: &'static str,
	}
	impl Localization {
		const fn get<const A: u8>() -> Self {
			match Language::from_index(A) {
				Language::Eng => Self {
					word: "Word",
					input_word: "Input word",
					solution: "Solution",
					solution_len: "Solution len",
					word_len: "Word len",
					points_float: "Points float",
					points: "Points",
				},
				Language::Ukr => Self {
					word: "Слово",
					input_word: "Введіть слово",
					solution: "Розв'язок",
					solution_len: "Довжина розв'язку",
					word_len: "Довжина слова",
					points_float: "Очків float",
					points: "Очків",
				}
			}
		}
	}

	fn get_word<const A: u8>(
		cli_args: &CliArgsPost,
		text_info_word: &str,
		text_prompt_word: &str,
		n: usize,
	) -> Word<A> {
		Word::new(&match &cli_args.word12 {
			Some(word12) => {
				let word_n_string = &word12[n-1];
				println!("{text_info_word} {n}: {word_n_string}");
				word_n_string.to_string()
			}
			None => prompt(&format!("{text_prompt_word} {n}: "))
		})
	}

	let loc = Localization::get::<A>();
	let get_word_lang = |n: usize| -> Word<A> {
		get_word(&cli_args, loc.word, loc.input_word, n)
	};
	let word1 = get_word_lang(1);
	let word2 = get_word_lang(2);
	let word2_len = word2.len();
	let solution = find_solution_st(word1, word2);
	println!("{}: {solution:#?}", loc.solution);
	let solution_len = solution.len();
	println!("{}: {solution_len}", loc.solution_len);
	println!("{} 2: {word2_len}", loc.word_len);
	let score_f = calc_score_f(word2_len, solution_len);
	println!("{}: {score_f}", loc.points_float);
	let score = calc_score(word2_len, solution_len);
	println!("{}: {score}", loc.points);
}



#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Allowed Actions aka Rules
enum Action {
	/* all features/rules:
	#[cfg(feature = "add")]
	#[cfg(feature = "remove")]
	#[cfg(feature = "replace")]
	#[cfg(feature = "swap")]
	#[cfg(feature = "discard")]
	#[cfg(feature = "copy")]
	#[cfg(feature = "take")]
	*/

	#[cfg(feature = "add")]
	Add { index: usize, char: char },

	#[cfg(feature = "remove")]
	Remove { index: usize },

	#[cfg(feature = "replace")]
	Replace { index: usize, char: char },

	// #[cfg(feature = "swap_one")]
	// SwapAtIndices { index1: usize, index2: usize },

	#[cfg(feature = "swap")]
	/// start and end indices are including
	Swap { index1s: usize, index1e: usize, index2s: usize, index2e: usize },

	#[cfg(feature = "discard")]
	/// start and end indices are including
	Discard { index_start: usize, index_end: usize },

	#[cfg(feature = "copy")]
	/// start and end indices are including
	Copy_ { index_start: usize, index_end: usize, index_insert: usize },

	#[cfg(feature = "take")]
	/// start and end indices are including
	Take { index_start: usize, index_end: usize },
}

impl Action {
	fn shift_indices_mut(&mut self, shift: usize) {
		use Action::*;
		match self {
			#[cfg(feature = "add")]
			Add { index, char: _ } => {
				*index += shift;
			}
			#[cfg(feature = "remove")]
			Remove { index } => {
				*index += shift;
			}
			#[cfg(feature = "replace")]
			Replace { index, char: _ } => {
				*index += shift;
			}
			#[cfg(feature = "swap")]
			Swap { index1s, index1e, index2s, index2e } => {
				*index1s += shift;
				*index1e += shift;
				*index2s += shift;
				*index2e += shift;
			}
			#[cfg(feature = "discard")]
			Discard { index_start, index_end } => {
				*index_start += shift;
				*index_end   += shift;
			}
			#[cfg(feature = "copy")]
			Copy_ { index_start, index_end, index_insert } => {
				*index_start  += shift;
				*index_end    += shift;
				*index_insert += shift;
			}
			#[cfg(feature = "take")]
			Take { index_start, index_end } => {
				*index_start += shift;
				*index_end   += shift;
			}
			#[allow(unreachable_patterns)] // reason: to test if works with `--no-default-features`
			_ => {}
		}
	}

	fn shifted_indices(mut self, shift: usize) -> Self {
		self.shift_indices_mut(shift);
		self
	}

	fn is_vain<const A: u8>(&self, word1: &Word<A>, word2: &Word<A>) -> bool {
		use Action::*;
		match self {
			// OPTIMIZATIONS 1:

			// // abcx -> zabc:
			// // 1. add{0,x} , remove{4} => 2 ops
			// // 2. remove{3} , add{0,x} => 2 ops
			// // 3. swap{0,2,3,3} => 1 op !!!
			// #[cfg(all(feature="add", feature="remove", feature="swap"))]
			// Add { .. } if word1.len() == word2.len() => true,

			#[cfg(feature = "add")]
			Add { .. } if word2.len() == 0 => true,
			#[cfg(feature = "remove")]
			Remove { .. } if word2.len() == 0 => false, // to avoid mistakes
			#[cfg(feature = "replace")]
			Replace { .. } if word2.len() == 0 => true,
			#[cfg(feature = "swap")]
			Swap { .. } if word2.len() == 0 => true,

			// TODO: more?

			_ => false
		}
	}

	/// `action_prev.is_stupid_before(action_next)`
	/// returns if `action_next` is stupid after `action_prev`.
	fn is_vain_with(&self, action_next: &Action) -> bool {
		use Action::*;
		match (self, action_next) {
			// OPTIMIZATIONS 2:

			#[cfg(feature = "replace")]
			(Replace { index: i1, .. }, Replace { index: i2, .. }) if i1 == i2 => true,

			#[cfg(all(feature="add", feature="replace"))]
			(Add { index: i1, .. }, Replace { index: i2, .. }) if i1 == i2 => true,

			#[cfg(all(feature="add", feature="remove"))]
			(Add { index: i1, .. }, Remove { index: i2 }) if i1 == i2 => true,

			_ => false
		}
	}
}



enum Language { Eng, Ukr }
impl Language {
	const ENG: u8 = 0;
	const UKR: u8 = 1;
	pub const fn from_index(lang_index: u8) -> Self {
		use Language::*;
		match lang_index {
			Self::ENG => Eng,
			Self::UKR => Ukr,
			_ => panic!()
		}
	}
	pub const fn get_alphabet(&self) -> &'static str {
		use Language::*;
		const ALPHABET_ENG: &str = "abcdefghijklmnopqrstuvwxyz";
		const ALPHABET_UKR: &str = "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя'";
		match self {
			Eng => ALPHABET_ENG,
			Ukr => ALPHABET_UKR,
		}
	}
	pub const fn get_alphabet_from_lang_index(lang_index: u8) -> &'static str {
		Self::from_index(lang_index).get_alphabet()
	}
}



type WordEng = Word<{Language::ENG}>;
type WordUkr = Word<{Language::UKR}>;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Word<const A: u8> {
	chars: Vec<char>,
}
impl<const A: u8> Word<A> {
	// const MAX_LEN: usize = 9;

	fn from(chars: &[char]) -> Self {
		// if chars.len() == 0 || chars.len() > Self::MAX_LEN { panic!() }
		// if chars.len() == 0 { panic!() }
		let alphabet = Language::get_alphabet_from_lang_index(A);
		assert!(chars.into_iter().all(|&c| alphabet.contains(c)));
		Self { chars: chars.to_vec() }
	}

	fn new(word_str: &str) -> Self {
		Self::from(&word_str.chars().collect::<Vec<char>>())
	}

	#[expect(unused)]
	fn to_string(&self) -> String {
		self.chars.iter().collect()
	}

	fn len(&self) -> usize {
		self.chars.len()
	}

	fn is_legal_action(&self, action: Action) -> bool {
		use Action::*;
		let self_len = self.len();
		match action {
			#[cfg(feature = "add")]
			Add { index, char: _ } => {
				// if self_len == Self::MAX_LEN { return false }
				if index > self_len { return false }
			}
			#[cfg(feature = "remove")]
			Remove { index } => {
				// if self_len == 1 { return false }
				if index >= self_len { return false }
			}
			#[cfg(feature = "replace")]
			Replace { index, char } => {
				if index >= self_len { return false }
				if self.chars[index] == char { return false }
			}
			#[cfg(feature = "swap")]
			Swap { index1s, index1e, index2s, index2e } => {
				if index1s >= self_len { return false }
				if index1e >= self_len { return false }
				if index2s >= self_len { return false }
				if index2e >= self_len { return false }
				if !(index1s <= index1e && index1e < index2s && index2s <= index2e) { return false }
			}
			#[cfg(feature = "discard")]
			Discard { index_start, index_end } => {
				if index_start >= self_len { return false }
				if index_end   >= self_len { return false }
				if !(index_start <= index_end) { return false }
			}
			#[cfg(feature = "copy")]
			Copy_ { index_start, index_end, index_insert } => {
				if index_start  >= self_len { return false }
				if index_end    >= self_len { return false }
				if index_insert >= self_len { return false }
				if !(index_start <= index_end) { return false }
			}
			#[cfg(feature = "take")]
			Take { index_start, index_end } => {
				if index_start >= self_len { return false }
				if index_end   >= self_len { return false }
				if !(index_start <= index_end) { return false }
			}
		}
		true
	}

	fn apply_action(&self, action: Action) -> Self {
		let mut self_ = self.clone();
		self_.apply_action_mut(action);
		self_
	}

	fn apply_action_mut(&mut self, action: Action) {
		use Action::*;
		if !self.is_legal_action(action) { panic!("self={self:?}\naction={action:?}") }
		// dbg!(action);
		match action {
			#[cfg(feature = "add")]
			Add { index, char } => {
				self.chars.insert(index, char);
			}
			#[cfg(feature = "remove")]
			Remove { index } => {
				let _ = self.chars.remove(index);
			}
			#[cfg(feature = "replace")]
			Replace { index, char } => {
				self.chars[index] = char;
			}
			#[cfg(feature = "swap")]
			Swap { index1s, index1e, index2s, index2e } => {
				let before = &self.chars[..index1s];
				let part_1 = &self.chars[index1s..=index1e];
				let middle = &self.chars[index1e+1..index2s];
				let part_2 = &self.chars[index2s..=index2e];
				let after  = &self.chars[index2e+1..];
				self.chars = [before, part_2, middle, part_1, after]
					.iter()
					.map(|p| p.iter().map(|&c| c))
					.flatten()
					.collect();
			}
			#[cfg(feature = "discard")]
			Discard { index_start, index_end } => {
				let _ = self.chars.drain(index_start..=index_end);
			}
			#[cfg(feature = "copy")]
			Copy_ { index_start, index_end, index_insert } => {
				let _ = self.chars.splice(
					index_insert..index_insert,
					self.chars[index_start..=index_end].to_vec()
				);
			}
			#[cfg(feature = "take")]
			Take { index_start, index_end } => {
				self.chars = self.chars[index_start..=index_end].to_vec();
			}
		}
	}

	fn all_actions_iter_by_coroutine(self) -> impl Iterator<Item=Action> {
		use Action::*;
		from_coroutine(#[coroutine] move || {
			let len = self.len();
			let alphabet = Language::get_alphabet_from_lang_index(A);

			#[cfg(feature = "remove")] // COMPLEXITY: L
			for index in 0..len {
				yield Remove { index }
			}

			#[cfg(feature = "take")] // COMPLEXITY: ~ L^2
			for index_start in 0..len {
				for index_end in index_start+1..len {
					yield Take { index_start, index_end }
				}
			}

			#[cfg(feature = "discard")] // COMPLEXITY: ~ L^2
			for index_start in 0..len {
				for index_end in index_start+1..len {
					yield Discard { index_start, index_end }
				}
			}

			#[cfg(feature = "replace")] // COMPLEXITY: L * A
			for index in 0..len {
				for char in alphabet.chars() {
					if self.chars[index] == char { continue }
					yield Replace { char, index }
				}
			}

			#[cfg(feature = "swap")] // COMPLEXITY: ~ L^4
			for index1s in 0..len {
				for index1e in index1s..len {
					for index2s in index1e+1..len {
						for index2e in index2s..len {
							yield Swap { index1s, index1e, index2s, index2e }
						}
					}
				}
			}

			#[cfg(feature = "add")] // COMPLEXITY: (L+1) * A
			for index in 0..=len {
				for char in alphabet.chars() {
					yield Add { index, char }
				}
			}

			#[cfg(feature = "copy")] // COMPLEXITY: ~ L^3
			for index_start in 0..len {
				for index_end in index_start+1..len {
					for index_insert in 0..=len {
						yield Copy_ { index_start, index_end, index_insert }
					}
				}
			}
		})
	}

	fn all_actions_vec(self) -> Vec<Action> {
		use Action::*;

		let len = self.len();
		let alphabet = Language::get_alphabet_from_lang_index(A);

		let mut actions_vec = vec![];

		#[cfg(feature = "remove")] // COMPLEXITY: L
		for index in 0..len {
			actions_vec.push(Remove { index });
		}

		#[cfg(feature = "take")] // COMPLEXITY: ~ L^2
		for index_start in 0..len {
			for index_end in index_start+1..len {
				actions_vec.push(Take { index_start, index_end });
			}
		}

		#[cfg(feature = "discard")] // COMPLEXITY: ~ L^2
		for index_start in 0..len {
			for index_end in index_start+1..len {
				actions_vec.push(Discard { index_start, index_end });
			}
		}

		#[cfg(feature = "replace")] // COMPLEXITY: L * A
		for index in 0..len {
			for char in alphabet.chars() {
				if self.chars[index] == char { continue }
				actions_vec.push(Replace { char, index });
			}
		}

		#[cfg(feature = "swap")] // COMPLEXITY: ~ L^4
		for index1s in 0..len {
			for index1e in index1s..len {
				for index2s in index1e+1..len {
					for index2e in index2s..len {
						actions_vec.push(Swap { index1s, index1e, index2s, index2e });
					}
				}
			}
		}

		#[cfg(feature = "add")] // COMPLEXITY: (L+1) * A
		for index in 0..=len {
			for char in alphabet.chars() {
				actions_vec.push(Add { index, char });
			}
		}

		#[cfg(feature = "copy")] // COMPLEXITY: ~ L^3
		for index_start in 0..len {
			for index_end in index_start+1..len {
				for index_insert in 0..=len {
					actions_vec.push(Copy_ { index_start, index_end, index_insert });
				}
			}
		}

		actions_vec.shrink_to_fit();
		actions_vec
	}

	fn all_actions_iter_by_gen_block(self) -> impl Iterator<Item=Action> {
		use Action::*;
		gen move { // edition2024 is required for this
			let len = self.len();
			let alphabet = Language::get_alphabet_from_lang_index(A);

			#[cfg(feature = "remove")] // COMPLEXITY: L
			for index in 0..len {
				yield Remove { index }
			}

			#[cfg(feature = "take")] // COMPLEXITY: ~ L^2
			for index_start in 0..len {
				for index_end in index_start+1..len {
					yield Take { index_start, index_end }
				}
			}

			#[cfg(feature = "discard")] // COMPLEXITY: ~ L^2
			for index_start in 0..len {
				for index_end in index_start+1..len {
					yield Discard { index_start, index_end }
				}
			}

			#[cfg(feature = "replace")] // COMPLEXITY: L * A
			for index in 0..len {
				for char in alphabet.chars() {
					if self.chars[index] == char { continue }
					yield Replace { char, index }
				}
			}

			#[cfg(feature = "swap")] // COMPLEXITY: ~ L^4
			for index1s in 0..len {
				for index1e in index1s..len {
					for index2s in index1e+1..len {
						for index2e in index2s..len {
							yield Swap { index1s, index1e, index2s, index2e }
						}
					}
				}
			}

			#[cfg(feature = "add")] // COMPLEXITY: (L+1) * A
			for index in 0..=len {
				for char in alphabet.chars() {
					yield Add { index, char }
				}
			}

			#[cfg(feature = "copy")] // COMPLEXITY: ~ L^3
			for index_start in 0..len {
				for index_end in index_start+1..len {
					for index_insert in 0..=len {
						yield Copy_ { index_start, index_end, index_insert }
					}
				}
			}
		}
	}

	fn dropped_at_index(&self, index: usize) -> Self {
		let mut self_ = self.clone();
		let _ = self_.chars.remove(index);
		self_
	}

	fn dropped_first(&self) -> Self {
		self.dropped_at_index(0)
	}

	fn dropped_last(&self) -> Self {
		self.dropped_at_index(self.len()-1)
	}
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PrefixSuffixLen { prefix_len: usize, suffix_len: usize }
impl From<(usize, usize)> for PrefixSuffixLen {
	fn from((prefix_len, suffix_len): (usize, usize)) -> Self {
		Self { prefix_len, suffix_len }
	}
}
impl Add<PrefixSuffixLen> for (usize, usize) {
	type Output = PrefixSuffixLen;
	fn add(self, rhs: PrefixSuffixLen) -> Self::Output {
		PrefixSuffixLen {
			prefix_len: self.0 + rhs.prefix_len,
			suffix_len: self.1 + rhs.suffix_len,
		}
	}
}
/// Returns number or commond letter in the begin and end of the word.
fn calc_common_prefix_and_suffix_len<const A: u8>(word1: &Word<A>, word2: &Word<A>) -> PrefixSuffixLen {
	if word1.len() == 0 || word2.len() == 0 { return (0, 0).into() }
	else if word1.chars.first() == word2.chars.first() { // if prefix
		return (1, 0) + calc_common_prefix_and_suffix_len(
			&word1.dropped_first(),
			&word2.dropped_first(),
		)
	}
	else if word1.chars.last() == word2.chars.last() { // if suffix
		return (0, 1) + calc_common_prefix_and_suffix_len(
			&word1.dropped_last(),
			&word2.dropped_last(),
		)
	}
	else {
		return (0, 0).into()
	}
}


fn find_solutions_st<const A: u8>(word_initial: Word<A>, word_target: Word<A>) -> Vec<Vec<Action>> {
	// dbg!(&word_initial, &word_target);
	if word_initial == word_target { return vec![vec![]] }
	let mut words: Vec<(Word<A>, Vec<Action>)> = vec![(word_initial, vec![])];
	let mut new_words: Vec<(Word<A>, Vec<Action>)> = vec![];
	let mut solutions: Vec<Vec<Action>> = vec![];
	let mut search_depth: u8 = 0;
	while solutions.is_empty() {
		search_depth += 1;
		// dbg!(search_depth);
		for (word, actions) in words.into_iter() {
			// dbg!(&word, &word_target, &actions);
			match calc_common_prefix_and_suffix_len(&word, &word_target) {
				PrefixSuffixLen { prefix_len: 0, suffix_len: 0 } => {
					for action in word.clone().all_actions_iter_by_coroutine() {
						// use optimizations 1:
						if action.is_vain(&word, &word_target) { continue }
						// use optimizations 2:
						if let Some(action_prev) = actions.last() && action_prev.is_vain_with(&action) { continue }
						let new_word = word.apply_action(action);
						// dbg!(&new_word);
						let new_actions = actions.clone().pushed_opt(action);
						if new_word == word_target { solutions.push(new_actions.clone()) }
						new_words.push((new_word, new_actions));
					}
				}
				PrefixSuffixLen { prefix_len, suffix_len } => {
					// ncp = non common part
					// dbg!(&word, &word_target);
					// dbg!(prefix_len, suffix_len);
					let word_ncp = Word::<A>::from(&word.chars[prefix_len..word.len()-suffix_len]);
					let word_target_ncp = Word::from(&word_target.chars[prefix_len..word_target.len()-suffix_len]);
					for solution in find_solutions_st(word_ncp, word_target_ncp) {
						// solutions.push(
						// 	actions
						// 		.clone()
						// 		.into_iter()
						// 		.chain(
						// 			solution
						// 				.iter()
						// 				.map(|a| a.shifted_indices(prefix_len))
						// 		)
						// 		.collect()
						// );

						let mut solution: Vec<Action> = solution
							.iter()
							.map(|a| a.shifted_indices(prefix_len))
							.collect();
						solution.shrink_to_fit();
						let mut actions = actions.clone();
						actions.extend(solution);
						actions.shrink_to_fit();
						solutions.push(actions);
					}
				}
			}
		}
		words = new_words;
		words.shrink_to_fit();
		new_words = vec![];
	}
	solutions
}

fn find_solution_st<const A: u8>(word_initial: Word<A>, word_target: Word<A>) -> Vec<Action> {
	find_solutions_st(word_initial, word_target)
		.into_iter()
		.min_by_key(|s| s.len())
		.unwrap()
}

fn find_solution_mt() {
	todo!()
}



fn calc_score_f(word2_len: usize, solution_len: usize) -> f64 {
	(word2_len as f64) / (solution_len as f64)
}

fn calc_score(word2_len: usize, solution_len: usize) -> u8 {
	calc_score_f(word2_len, solution_len).round() as u8
}



#[cfg(test)]
mod tests {
	use super::*;

	mod find_solution {
		use super::*;

		#[test]
		fn trivial() {
			assert_eq!(
				Vec::<Action>::new(),
				find_solution_st(WordEng::new("foobar"), WordEng::new("foobar"))
			)
		}

		#[cfg(feature = "add")]
		mod add {
			use super::*;
			use Action::Add;
			#[test]
			fn b() {
				assert_eq!(
					vec![Add { index: 0, char: 'x' }],
					find_solution_st(WordEng::new("foobar"), WordEng::new("xfoobar"))
				)
			}
			#[test]
			fn bb() {
				assert_eq!(
					vec![
						Add { index: 0, char: 'x' },
						Add { index: 0, char: 'y' },
					],
					find_solution_st(WordEng::new("foobar"), WordEng::new("yxfoobar"))
				)
			}
			#[test]
			fn bbb() {
				assert_eq!(
					vec![
						Add { index: 0, char: 'x' },
						Add { index: 0, char: 'y' },
						Add { index: 0, char: 'z' },
					],
					find_solution_st(WordEng::new("foobar"), WordEng::new("zyxfoobar"))
				)
			}
			#[test]
			fn e() {
				assert_eq!(
					vec![Add { index: 6, char: 'x' }],
					find_solution_st(WordEng::new("foobar"), WordEng::new("foobarx"))
				)
			}
			#[test]
			fn ee() {
				assert_eq!(
					vec![
						Add { index: 6, char: 'x' },
						Add { index: 7, char: 'y' },
					],
					find_solution_st(WordEng::new("foobar"), WordEng::new("foobarxy"))
				)
			}
			#[test]
			fn eee() {
				assert_eq!(
					vec![
						Add { index: 6, char: 'x' },
						Add { index: 7, char: 'y' },
						Add { index: 8, char: 'z' },
					],
					find_solution_st(WordEng::new("foobar"), WordEng::new("foobarxyz"))
				)
			}
			#[test]
			fn m() {
				assert_eq!(
					vec![Add { index: 3, char: 'x' }],
					find_solution_st(WordEng::new("foobar"), WordEng::new("fooxbar"))
				)
			}
			#[test]
			fn mm() {
				assert_eq!(
					vec![
						Add { index: 3, char: 'x' },
						Add { index: 4, char: 'y' },
					],
					find_solution_st(WordEng::new("foobar"), WordEng::new("fooxybar"))
				)
			}
			#[test]
			fn mmm() {
				assert_eq!(
					vec![
						Add { index: 3, char: 'x' },
						Add { index: 4, char: 'y' },
						Add { index: 5, char: 'z' },
					],
					find_solution_st(WordEng::new("foobar"), WordEng::new("fooxyzbar"))
				)
			}
		}

		#[cfg(feature = "remove")]
		mod remove {
			use super::*;
			use Action::Remove;
			#[test]
			fn b() {
				assert_eq!(
					vec![Remove { index: 0 }],
					find_solution_st(WordEng::new("foobar"), WordEng::new("oobar"))
				)
			}
			#[ignore = "discard solves it better"]
			#[test]
			fn bb() {
				assert_eq!(
					vec![
						Remove { index: 0 },
						Remove { index: 0 },
					],
					find_solution_st(WordEng::new("foobar"), WordEng::new("obar"))
				)
			}
			#[ignore = "discard solves it better"]
			#[test]
			fn bbb() {
				assert_eq!(
					vec![
						Remove { index: 0 },
						Remove { index: 0 },
						Remove { index: 0 },
					],
					find_solution_st(WordEng::new("foobar"), WordEng::new("bar"))
				)
			}
			#[test]
			fn e() {
				assert_eq!(
					vec![Remove { index: 5 }],
					find_solution_st(WordEng::new("foobar"), WordEng::new("fooba"))
				)
			}
			#[ignore = "discard solves it better"]
			#[test]
			fn ee() {
				let expected_solutions = vec![
					vec![
						Remove { index: 5 },
						Remove { index: 4 },
					],
					vec![
						Remove { index: 4 },
						Remove { index: 4 },
					],
				];
				let actual_solution = find_solution_st(WordEng::new("foobar"), WordEng::new("foob"));
				dbg!(&expected_solutions, &actual_solution);
				assert!(expected_solutions.contains(&actual_solution))
			}
			#[ignore = "discard solves it better"]
			#[test]
			fn eee() {
				let expected_solutions = vec![
					vec![
						Remove { index: 5 },
						Remove { index: 4 },
						Remove { index: 3 },
					],
					vec![
						Remove { index: 3 },
						Remove { index: 3 },
						Remove { index: 3 },
					],
				];
				let actual_solution = find_solution_st(WordEng::new("foobar"), WordEng::new("foo"));
				dbg!(&expected_solutions, &actual_solution);
				assert!(expected_solutions.contains(&actual_solution))
			}
		}

		#[cfg(feature = "replace")]
		mod replace {
			use super::*;
			use Action::Replace;
			#[test]
			fn b() {
				assert_eq!(
					vec![Replace { index: 0, char: 'x' }],
					find_solution_st(WordEng::new("foobar"), WordEng::new("xoobar"))
				)
			}
			#[test]
			fn e() {
				assert_eq!(
					vec![Replace { index: 5, char: 'x' }],
					find_solution_st(WordEng::new("foobar"), WordEng::new("foobax"))
				)
			}
			#[test]
			fn m() {
				assert_eq!(
					vec![Replace { index: 2, char: 'x' }],
					find_solution_st(WordEng::new("foobar"), WordEng::new("foxbar"))
				)
			}
		}

		#[cfg(feature = "swap")]
		mod swap {
			use super::*;
			use Action::Swap;
			#[test]
			fn foobar_barfoo() {
				assert_eq!(
					vec![Swap { index1s: 0, index1e: 2, index2s: 3, index2e: 5 }],
					find_solution_st(WordEng::new("foobar"), WordEng::new("barfoo"))
				)
			}
			#[test]
			fn abcfoodefbarxyz_abcbardeffooxyz() {
				assert_eq!(
					vec![Swap { index1s: 3, index1e: 5, index2s: 9, index2e: 11 }],
					find_solution_st(WordEng::new("abcfoodefbarxyz"), WordEng::new("abcbardeffooxyz"))
				)
			}
		}

		#[cfg(feature = "discard")]
		mod discard {
			use super::*;
			use Action::Discard;
			#[test]
			fn b2() {
				assert_eq!(
					vec![Discard { index_start: 0, index_end: 1 }],
					find_solution_st(WordEng::new("foobar"), WordEng::new("obar"))
				)
			}
			#[test]
			fn b3() {
				assert_eq!(
					vec![Discard { index_start: 0, index_end: 2 }],
					find_solution_st(WordEng::new("foobar"), WordEng::new("bar"))
				)
			}
			#[test]
			fn e2() {
				assert_eq!(
					vec![Discard { index_start: 4, index_end: 5 }],
					find_solution_st(WordEng::new("foobar"), WordEng::new("foob"))
				)
			}
			#[test]
			fn e3() {
				assert_eq!(
					vec![Discard { index_start: 3, index_end: 5 }],
					find_solution_st(WordEng::new("foobar"), WordEng::new("foo"))
				)
			}
			#[test]
			fn m2() {
				assert_eq!(
					vec![Discard { index_start: 2, index_end: 3 }],
					find_solution_st(WordEng::new("foobar"), WordEng::new("foar"))
				)
			}
			#[test]
			fn m4() {
				assert_eq!(
					vec![Discard { index_start: 1, index_end: 4 }],
					find_solution_st(WordEng::new("foobar"), WordEng::new("fr"))
				)
			}
		}

		#[cfg(feature = "copy")]
		mod copy {
			use super::*;
			use Action::Copy_;
			#[test]
			fn foo_foofoo() {
				let expected_solutions = [
					vec![Copy_ { index_start: 0, index_end: 2, index_insert: 0 }],
					vec![Copy_ { index_start: 0, index_end: 2, index_insert: 3 }],
				];
				let actual_solution = find_solution_st(
					WordEng::new("foo"),
					WordEng::new("foofoo")
				);
				dbg!(&expected_solutions, &actual_solution);
				assert!(expected_solutions.contains(&actual_solution))
			}
			#[test]
			fn foobar_foobarfoo() {
				assert_eq!(
					vec![Copy_ { index_start: 0, index_end: 2, index_insert: 6 }],
					find_solution_st(WordEng::new("foobar"), WordEng::new("foobarfoo"))
				)
			}
			#[test]
			fn foobar_barfoobar() {
				assert_eq!(
					vec![Copy_ { index_start: 3, index_end: 5, index_insert: 0 }],
					find_solution_st(WordEng::new("foobar"), WordEng::new("barfoobar"))
				)
			}
		}

		#[cfg(feature = "take")]
		mod take {
			use super::*;
			use Action::Take;
			#[test]
			fn foobar_foo() {
				assert_eq!(
					vec![Take { index_start: 3, index_end: 5 }],
					find_solution_st(WordEng::new("foobarxyz"), WordEng::new("bar"))
				)
			}
		}
	}

	mod calc_common_prefix_and_suffix_len {
		use super::*;
		#[test]
		fn abcxyzdefgh_abcvdefgh() {
			assert_eq!(
				PrefixSuffixLen { prefix_len: 3, suffix_len: 5 },
				calc_common_prefix_and_suffix_len(
					&WordEng::new("abcxyzdefgh"),
					&WordEng::new("abcvdefgh")
				)
			)
		}
		#[test]
		fn kzko_ko() {
			let expected_solutions = [
				PrefixSuffixLen { prefix_len: 0, suffix_len: 2 },
				PrefixSuffixLen { prefix_len: 1, suffix_len: 1 },
			];
			let actual_solution = calc_common_prefix_and_suffix_len(
				&WordEng::new("kzko"),
				&WordEng::new("ko")
			);
			dbg!(expected_solutions, actual_solution);
			assert!(expected_solutions.contains(&actual_solution))
		}
		#[test]
		fn kzz_k() {
			assert_eq!(
				PrefixSuffixLen { prefix_len: 1, suffix_len: 0 },
				calc_common_prefix_and_suffix_len(
					&WordEng::new("kzz"),
					&WordEng::new("k")
				)
			)
		}
		#[test]
		fn zzk_k() {
			assert_eq!(
				PrefixSuffixLen { prefix_len: 0, suffix_len: 1 },
				calc_common_prefix_and_suffix_len(
					&WordEng::new("zzk"),
					&WordEng::new("k")
				)
			)
		}
	}
}

