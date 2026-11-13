//! Show-o's MaskGIT step in integers, so the chip's firmware and the host give the same
//! codes: each masked place draws a code from its scores, as a softmax would, and gets a
//! confidence, its log-probability plus Gumbel noise that dies down; the least sure stay
//! masked for the next step. Exponentials and logarithms are base 2, from tables, in fixed
//! point; it needs no heap, so the same code runs in the firmware.
#![no_std]

/// Fractional bits of the logarithms and of the confidences.
pub const FRACTION: u32 = 16;
/// log2(e) and ln(2), with `FRACTION` fractional bits.
const LOG2_E: i64 = 94548;
const LN_2: i64 = 45426;
/// Weights of the softmax have this many fractional bits.
const WEIGHT_BITS: u32 = 24;
/// Below 2^-30 of the best, a code's weight is nothing.
const FLOOR: i64 = -30 << FRACTION;

/// `2^(-i/256)`, with `WEIGHT_BITS` fractional bits.
const EXP2: [u32; 256] = [
    16777216, 16731851, 16686609, 16641490, 16596492, 16551616, 16506861, 16462228, 16417715, 16373322, 16329050, 16284897,
    16240863, 16196949, 16153153, 16109476, 16065917, 16022476, 15979152, 15935945, 15892855, 15849882, 15807025, 15764283,
    15721658, 15679147, 15636752, 15594471, 15552304, 15510252, 15468313, 15426487, 15384775, 15343175, 15301688, 15260313,
    15219050, 15177899, 15136859, 15095929, 15055111, 15014403, 14973805, 14933316, 14892937, 14852668, 14812507, 14772455,
    14732511, 14692675, 14652947, 14613326, 14573813, 14534406, 14495106, 14455912, 14416824, 14377841, 14338964, 14300193,
    14261526, 14222963, 14184505, 14146151, 14107901, 14069754, 14031710, 13993769, 13955931, 13918195, 13880561, 13843028,
    13805598, 13768268, 13731039, 13693911, 13656884, 13619956, 13583129, 13546401, 13509772, 13473242, 13436812, 13400479,
    13364245, 13328109, 13292070, 13256129, 13220286, 13184539, 13148888, 13113335, 13077877, 13042515, 13007249, 12972078,
    12937002, 12902021, 12867135, 12832343, 12797645, 12763041, 12728530, 12694113, 12659789, 12625557, 12591419, 12557372,
    12523418, 12489555, 12455784, 12422104, 12388516, 12355018, 12321610, 12288294, 12255067, 12221930, 12188882, 12155924,
    12123055, 12090275, 12057584, 12024981, 11992466, 11960039, 11927700, 11895448, 11863283, 11831206, 11799215, 11767310,
    11735492, 11703760, 11672114, 11640553, 11609078, 11577687, 11546382, 11515161, 11484025, 11452973, 11422004, 11391120,
    11360319, 11329601, 11298967, 11268415, 11237946, 11207559, 11177254, 11147032, 11116891, 11086831, 11056853, 11026956,
    10997140, 10967404, 10937749, 10908174, 10878679, 10849263, 10819928, 10790671, 10761494, 10732395, 10703375, 10674434,
    10645571, 10616786, 10588079, 10559449, 10530897, 10502422, 10474024, 10445703, 10417458, 10389290, 10361198, 10333182,
    10305242, 10277377, 10249587, 10221873, 10194234, 10166669, 10139179, 10111763, 10084422, 10057154, 10029960, 10002839,
    9975792, 9948818, 9921917, 9895089, 9868333, 9841650, 9815039, 9788499, 9762032, 9735636, 9709311, 9683058,
    9656875, 9630764, 9604722, 9578752, 9552851, 9527021, 9501261, 9475570, 9449948, 9424396, 9398913, 9373499,
    9348154, 9322877, 9297668, 9272528, 9247455, 9222451, 9197514, 9172644, 9147842, 9123107, 9098438, 9073837,
    9049301, 9024833, 9000430, 8976093, 8951823, 8927617, 8903477, 8879403, 8855394, 8831449, 8807569, 8783754,
    8760003, 8736317, 8712694, 8689136, 8665641, 8642209, 8618841, 8595536, 8572295, 8549116, 8525999, 8502945,
    8479954, 8457025, 8434157, 8411352,
];

/// `log2(1 + i/256)`, with `FRACTION` fractional bits.
const LOG2: [u32; 256] = [
    0, 369, 736, 1102, 1466, 1829, 2190, 2551, 2909, 3267, 3623, 3978,
    4331, 4683, 5034, 5384, 5732, 6079, 6425, 6769, 7112, 7454, 7795, 8134,
    8473, 8810, 9146, 9480, 9814, 10146, 10477, 10807, 11136, 11464, 11791, 12116,
    12440, 12764, 13086, 13407, 13727, 14046, 14363, 14680, 14996, 15310, 15624, 15937,
    16248, 16559, 16868, 17177, 17484, 17791, 18096, 18401, 18704, 19007, 19308, 19609,
    19909, 20207, 20505, 20802, 21098, 21393, 21687, 21980, 22272, 22564, 22854, 23144,
    23433, 23720, 24007, 24293, 24579, 24863, 25146, 25429, 25711, 25992, 26272, 26551,
    26830, 27108, 27384, 27660, 27936, 28210, 28484, 28757, 29029, 29300, 29571, 29840,
    30109, 30378, 30645, 30912, 31178, 31443, 31707, 31971, 32234, 32496, 32758, 33019,
    33279, 33538, 33797, 34055, 34312, 34569, 34825, 35080, 35334, 35588, 35841, 36094,
    36346, 36597, 36847, 37097, 37346, 37595, 37842, 38090, 38336, 38582, 38827, 39072,
    39316, 39559, 39802, 40044, 40286, 40527, 40767, 41006, 41246, 41484, 41722, 41959,
    42196, 42432, 42667, 42902, 43137, 43370, 43603, 43836, 44068, 44300, 44530, 44761,
    44990, 45220, 45448, 45676, 45904, 46131, 46357, 46583, 46809, 47034, 47258, 47482,
    47705, 47928, 48150, 48372, 48593, 48813, 49034, 49253, 49472, 49691, 49909, 50127,
    50344, 50560, 50776, 50992, 51207, 51422, 51636, 51850, 52063, 52276, 52488, 52700,
    52911, 53122, 53332, 53542, 53751, 53960, 54169, 54377, 54584, 54791, 54998, 55204,
    55410, 55615, 55820, 56025, 56229, 56432, 56635, 56838, 57040, 57242, 57443, 57644,
    57845, 58045, 58245, 58444, 58643, 58841, 59039, 59237, 59434, 59631, 59827, 60023,
    60219, 60414, 60609, 60803, 60997, 61190, 61384, 61576, 61769, 61961, 62152, 62343,
    62534, 62725, 62915, 63104, 63294, 63483, 63671, 63859, 64047, 64234, 64421, 64608,
    64794, 64980, 65166, 65351,
];

/// A small random generator, xorshift, the same on the chip and the host.
#[derive(Clone, Copy, Debug)]
pub struct Rng(pub u32);

impl Rng {
    pub fn next_word(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
}

/// `2^(t / 2^FRACTION)` for `t <= 0`, with `WEIGHT_BITS` fractional bits.
fn exp2(t: i64) -> u64 {
    let down = (-t) as u64;
    let (whole, part) = (down >> FRACTION, (down >> (FRACTION - 8)) & 255);
    if whole >= 63 {
        return 0;
    }
    EXP2[part as usize] as u64 >> whole
}

/// `log2(x)` of `x > 0`, with `FRACTION` fractional bits.
fn log2(x: u64) -> i64 {
    let top = 63 - x.leading_zeros() as i64;
    let next = if top >= 8 { (x >> (top - 8)) & 255 } else { (x << (8 - top)) & 255 };
    (top << FRACTION) + LOG2[next as usize] as i64
}

/// One masked place's draw from its scores: `score(c)` is code `c`'s mantissa, the
/// scores sharing `exponent`, with mantissas of `bits` bits, so code `c`'s score is
/// `score(c) · 2^(exponent − bits + 1)`. Gives the code drawn and its confidence: its
/// natural log-probability plus `temperature` times Gumbel noise, both with `FRACTION`
/// fractional bits, `temperature` too.
pub fn draw(codes: u32, score: &dyn Fn(u32) -> i32, exponent: i32, bits: u32, temperature: i64, rng: &mut Rng) -> (u32, i64) {
    let best = (0..codes).map(score).max().unwrap_or(0) as i64;
    // Each score less the best, in base-2 logarithms.
    let shift = exponent - bits as i32 + 1;
    let log2_of = |c: u32| -> i64 {
        let natural = (score(c) as i64 - best) * LOG2_E;
        let t = if shift >= 0 { natural << shift.min(30) } else { natural >> (-shift).min(62) };
        t.max(FLOOR)
    };
    let total: u64 = (0..codes).map(|c| exp2(log2_of(c))).sum();
    // Up to 2^13 codes of 2^24 each add up past 2^32: the product needs 128 bits.
    let pick = ((rng.next_word() as u128 * total as u128) >> 32) as u64;
    let mut sum = 0;
    let mut code = codes - 1;
    for c in 0..codes {
        sum += exp2(log2_of(c));
        if sum > pick {
            code = c;
            break;
        }
    }
    let log2_p = log2_of(code) - (log2(total.max(1)) - ((WEIGHT_BITS as i64) << FRACTION));
    // Gumbel noise, −ln(−ln u), for u in (0, 1).
    let u = (rng.next_word() >> 8) as u64 | 1;
    let minus_ln_u = (((24 << FRACTION) - log2(u)) * LN_2) >> FRACTION;
    let gumbel = -(((log2(minus_ln_u.max(1) as u64) - ((FRACTION as i64) << FRACTION)) * LN_2) >> FRACTION);
    (code, ((log2_p * LN_2) >> FRACTION) + ((temperature * gumbel) >> FRACTION))
}

/// How many places stay masked after a step: the schedule's `remaining`, but fewer than
/// the `unknown` ones and at least one.
pub fn masked(remaining: i64, unknown: usize) -> usize {
    remaining.min(unknown as i64 - 1).max(1) as usize
}

/// The confidence below which a place stays masked: the `masked`-th smallest of all of
/// them, counting from zero.
pub fn cut(confidences: &[i64], masked: usize) -> i64 {
    let index = masked.min(confidences.len() - 1);
    // The value with exactly `index` smaller ones, ties counted once.
    let mut best = i64::MAX;
    for (i, c) in confidences.iter().enumerate() {
        let smaller = confidences.iter().filter(|o| *o < c).count();
        let equal_before = confidences[..i].iter().filter(|o| *o == c).count();
        if smaller + equal_before == index {
            best = *c;
        }
    }
    best
}
