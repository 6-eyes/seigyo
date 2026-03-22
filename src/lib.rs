pub trait Float:
    core::ops::Neg<Output = Self>
    + core::ops::Div<Output = Self>
    + core::ops::DivAssign
    + core::ops::Mul<Output = Self>
    + core::ops::MulAssign
    + core::ops::Add<Output = Self>
    + core::ops::AddAssign
    + core::ops::Sub<Output = Self>
    + core::ops::SubAssign
    + core::iter::Sum<Self>
    + core::cmp::PartialOrd
    + core::fmt::Display
    + core::fmt::Debug
    + PartialEq
    + Copy
    + Sized
    + Default
    // compare to real complex numbers
    + PartialEq<Complex<Self>>
    + core::ops::Add<Complex<Self>, Output = Complex<Self>>
    + core::ops::Sub<Complex<Self>, Output = Complex<Self>>
{
    /// abstracting square root from floating types
    fn sqrt(self) -> Self;

    /// abstracting powi from floating types
    fn powi(self, n: i32) -> Self;

    /// abstracting round from floating types
    fn round(self) -> Self;

    /// abstracting abs from floating types
    fn abs(self) -> Self;

    /// abstracting arctan2 from floating types
    fn atan2(self, other: Self) -> Self;

    fn min(self, other: Self) -> Self {
        if self > other { other } else { self }
    }

    // returns the max of the two numbers
    fn max(self, other: Self) -> Self {
        if self < other { other } else { self }
    }

    /// the minimum tolerated value. any value less than this will be considered zero.
    fn tolerance() -> Self;

    /// abstracting sin from floating types
    fn sin(self) -> Self;

    /// abstracting cos from floating types
    fn cos(self) -> Self;

    /// type conversion from f64 to Self
    fn from_f64(v: f64) -> Self;

    /// type conversion from f32 to Self
    fn from_f32(v: f32) -> Self;
}

impl Float for f32 {
    #[inline(always)]
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    
    #[inline(always)]
    fn powi(self, n: i32) -> Self {
        self.powi(n)
    }

    #[inline(always)]
    fn round(self) -> Self {
        self.round()
    }

    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }

    #[inline(always)]
    fn atan2(self, other: Self) -> Self {
        self.atan2(other)    
    }

    #[inline(always)]
    fn tolerance() -> Self {
        const TOLERANCE_F32: f32 = 1e-6;
        TOLERANCE_F32
    }

    #[inline(always)]
    fn sin(self) -> Self {
       self.sin() 
    }

    #[inline(always)]
    fn cos(self) -> Self {
        self.cos()
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        if self > other { other } else { self }
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        if self < other { other } else { self }
    }

    #[inline(always)]
    fn from_f32(v: f32) -> Self {
        v
    }

    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v as Self
    }
}

impl Float for f64 {
    #[inline(always)]
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    
    #[inline(always)]
    fn powi(self, n: i32) -> Self {
        self.powi(n)
    }

    #[inline(always)]
    fn round(self) -> Self {
        self.round()
    }

    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }

    #[inline(always)]
    fn atan2(self, other: Self) -> Self {
        self.atan2(other)    
    }

    #[inline(always)]
    fn tolerance() -> Self {
        const TOLERANCE_F64: f64 = 1e-12;
        TOLERANCE_F64
    }

    #[inline(always)]
    fn sin(self) -> Self {
        self.sin()
    }

    #[inline(always)]
    fn cos(self) -> Self {
        self.cos()
    }

    #[inline(always)]
    fn from_f32(v: f32) -> Self {
        v as Self
    }

    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v
    }
}

/// Defines a complex number
#[derive(Debug, Clone, Copy)]
pub struct Complex<T: Float = f64>(T, T);

impl<T: Float> Default for Complex<T> {
    fn default() -> Self {
        let zero = T::default();

        Self(zero, zero)
    }
}

impl<T: Float> Complex<T> {
    /// ### New
    /// creates a new complex number provided the real and imaginary values
    pub fn new(real: T, imaginary: T) -> Self {
        Self(real, imaginary)
    }

    /// returns the real part of the complex number
    #[inline(always)]
    pub fn real(&self) -> T {
        self.0
    }
    
    /// returns the imaginary part of the complex number
    #[inline(always)]
    pub fn imaginary(&self) -> T {
        self.1
    }
    
    /// ### Conjugate
    /// - Returns the conjugate of the complex number
    /// - Negates the imaginary part of the complex number.
    /// ```rust
    /// use seigyo::Complex;
    ///
    /// let complex = Complex::from((3., 4.));
    /// assert_eq!(complex.conjugate(), Complex::from((3., -4.)))
    /// ```
    pub fn conjugate(mut self) -> Self {
        self.1 = -self.1;
        self
    }

    /// ### Argument
    /// Returns the argument of the complex number
    /// ```rust
    /// use seigyo::Complex;
    /// use core::f32::consts::PI;
    /// 
    /// let complex = Complex::from((1., 1.));
    /// assert_eq!(complex.arg(), PI / 4.);
    /// ```
    pub fn arg(&self) -> T {
        self.1.atan2(self.0)
    }
    
    /// ### Magnitude
    /// Calculates the magnitude of the complex number
    /// Magnitude of $a + ib$ given by:$$a^2 + b^2$$
    /// 
    /// **Note:** this method returns the real/imaginary value if the other one is below the tolerance.
    /// ```rust
    /// use seigyo::Complex;
    ///
    /// let complex = Complex::from((3., 4.));
    /// assert_eq!(complex.magnitude(), 5.);
    /// ```
    #[inline(always)]
    pub fn magnitude(&self) -> T {
        if self.is_real() {
            self.0.abs()
        }
        else if self.is_imaginary()  {
            self.1.abs()
        }
        else {
            self.normsq().sqrt()
        }
    }

    /// returns the square of the norm
    /// #### Motivation
    /// - this can be used in multiple place where the equation demands square of a norm.
    /// - this can be used to avoid the step of calculating square root and then calculating the square again.
    #[inline(always)]
    pub fn normsq(&self) -> T {
        self.0.powi(2) + self.1.powi(2)
    }

    /// rounds of the real and imaginary parts of the complex number
    fn round(self) -> Self {
        Self(self.0.round(), self.1.round())
    }

    /// normalizes the [Complex] number
    /// ```rust
    /// use seigyo::Complex;
    /// 
    /// let complex = Complex::new(3., 4.);
    /// assert_eq!(complex.normalize(), Complex::new(0.6, 0.8))
    /// ```
    #[inline(always)]
    pub fn normalize(self) -> Self {
        match (self.0.abs() < T::tolerance(), self.1.abs() < T::tolerance()) {
            (true, true) => self,
            (false, true) => self / self.0,
            (true, false) => self / self.1,
            (false, false) => self / self.normsq().sqrt(),
        }
    }

    /// ### Is real
    /// Returns `true` is the complex number has an imaginary value lesser than the tolerance regardless of the real value.
    /// ```rust
    /// use seigyo::Complex;
    ///
    /// let complex = Complex::from(2.);
    /// assert_eq!(complex.is_real(), true);
    /// ```
    #[inline(always)]
    pub fn is_real(&self) -> bool {
        self.1.abs() < T::tolerance()
    }

    /// ### Is imaginary
    /// Returns `true` is the complex number has an real value lesser than the tolerance regardless of the imaginary value.
    /// ```rust
    /// use seigyo::Complex;
    ///
    /// let complex = Complex::from((0., 2.));
    /// assert_eq!(complex.is_imaginary(), true);
    /// ```
    #[inline(always)]
    pub fn is_imaginary(&self) -> bool {
        self.0.abs() < T::tolerance()
    }

    /// ### Is zero
    /// Returns `true` if both the real and imaginary parts have values lesser than the tolerance
    /// ```rust
    /// use seigyo::Complex;
    ///
    /// let complex = Complex::from((0., 0.));
    /// assert_eq!(complex.is_zero(), true);
    /// ```
    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        self.is_imaginary() && self.is_real()
    }

    #[inline(always)]
    fn is_one(&self) -> bool {
        self.is_real() && (self.0 - T::from_f32(1.)).abs() < T::tolerance()
    }

    /// ### Dot product
    /// For two complex numbers $(a + ib)$ and $(c + id)$, the dot product is given by $ac + bd$.
    pub fn dot(&self, other: Self) -> T {
        self.0 * other.0 + self.1 * other.1
    }
}

/// ## Divide
impl<T: Float> core::ops::Div for Complex<T> {
    type Output = Self;
    
    /// ```rust
    /// use seigyo::Complex;
    ///
    /// let a = Complex::from((1., 2.));
    /// let b = Complex::from((3., 4.));
    ///  
    /// assert_eq!(a / b, Complex::from((-0.2, 0.4)))
    /// ```
    fn div(mut self, other: Complex<T>) -> Self::Output {
        self /= other;
        self
    }
}

impl<T: Float> core::ops::Div<T> for Complex<T> {
    type Output = Self;

    /// ```rust
    /// use seigyo::Complex;
    ///
    /// let a = Complex::from((1., 2.));
    ///
    /// assert_eq!(a / 2., Complex::from((0.5, 1.)))
    /// ```
    fn div(mut self, rhs: T) -> Self::Output {
        self /= rhs;
        self
    }
}

/// ## Divide assign
/// divide assign complex number by a complex number
impl<T: Float> core::ops::DivAssign for Complex<T> {
    /// ```rust
    /// use seigyo::Complex;
    ///
    /// let mut a = Complex::from((1., 2.));
    /// let b = Complex::from((3., 4.));
    ///  
    /// a /= b;
    /// assert_eq!(a, Complex::from((-0.2, 0.4)))
    /// ```
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        let mag = rhs.normsq();
        *self = Self((self.0 * rhs.0 - self.1 * rhs.1) / mag, (self.0 * rhs.1 + self.1 * rhs.0) / mag);
    }
}

/// divide assign complex number by a float
impl<T: Float> core::ops::DivAssign<T> for Complex<T> {
    /// ```rust
    /// use seigyo::Complex;
    ///
    /// let mut a = Complex::from((1., 2.));
    ///  
    /// a /= 2.;
    /// assert_eq!(a, Complex::from((0.5, 1.)))
    /// ```
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        self.0 /= rhs;
        self.1 /= rhs;
    }
}

/// ## Multiply
/// multiply complex number by a complex number
impl<T: Float> core::ops::Mul for Complex<T> {
    type Output = Self;
    
    /// ```rust
    /// use seigyo::Complex;
    ///
    /// let a = Complex::from((1., 2.));
    /// let b = Complex::from((3., 4.));
    /// 
    /// assert_eq!(a * b, Complex::from((-5., 10.)))
    /// ```
    fn mul(mut self, other: Complex<T>) -> Self::Output {
        self *= other;
        self
    }
}

/// multiply complex number by a float
impl<T: Float> core::ops::Mul<T> for Complex<T> {
    type Output = Self;
    
    /// ```rust
    /// use seigyo::Complex;
    ///
    /// let a = Complex::from((1., 2.));
    /// 
    /// assert_eq!(a * 2., Complex::from((2., 4.)))
    /// ```
    fn mul(mut self, other: T) -> Self::Output {
       self *= other;
       self
    }
}

/// multiply complex number by a referenced complex number
impl<T: Float> core::ops::Mul<&Complex<T>> for Complex<T> {
    type Output = Complex<T>;

    /// ```rust
    /// use seigyo::Complex;
    ///
    /// let a = Complex::from((1., 2.));
    /// let b = Complex::from((3., 4.));
    /// 
    /// assert_eq!(a * &b, Complex::from((-5., 10.)))
    /// ```
    fn mul(mut self, rhs: &Complex<T>) -> Self::Output {
        self *= rhs;
        self
    }
}

/// ### Multiply by a matrix
/// multiply a complex number by a matrix
impl<T: Float, const R: usize, const C: usize> core::ops::Mul<Matrix<R, C, T>> for Complex<T> {
    type Output = Matrix<R, C, T>;

    /// ### Example
    ///
    /// ```rust
    /// use seigyo::{Matrix, Complex};
    ///
    /// let a = Matrix::from([[1., 8., 3.], [9., 4., 5.], [6., 2., 7.]]);
    /// assert_eq!(Complex::from(2.) * a, Matrix::from([[2., 16., 6.], [18., 8., 10.], [12., 4., 14.]]));
    /// ```
    fn mul(self, mut rhs: Matrix<R, C, T>) -> Self::Output {
        rhs *= self;
        rhs
    } 
}

/// ## Multiply assign
/// complex number multiplied by a complex number
impl<T: Float> core::ops::MulAssign for Complex<T> {
    /// ```rust
    /// use seigyo::Complex;
    ///
    /// let mut a = Complex::from((1., 2.));
    /// let b = Complex::from((3., 4.));
    /// 
    /// a *= b;
    ///
    /// assert_eq!(a, Complex::from((-5., 10.)))
    /// ```
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = Self(self.0 * rhs.0 - self.1 * rhs.1, self.0 * rhs.1 + self.1 * rhs.0);
    }
}

/// complex number multiplied by a referenced complex number
impl<T: Float> core::ops::MulAssign<&Complex<T>> for Complex<T> {
    /// ```rust
    /// use seigyo::Complex;
    ///
    /// let mut a = Complex::from((1., 2.));
    /// let b = Complex::from((3., 4.));
    /// 
    /// a *= &b;
    ///
    /// assert_eq!(a, Complex::from((-5., 10.)))
    /// ```
    #[inline]
    fn mul_assign(&mut self, rhs: &Complex<T>) {
        *self = Self(self.0 * rhs.0 - self.1 * rhs.1, self.0 * rhs.1 + self.1 * rhs.0);
    }
}

/// complex number multiplied by a floating number
impl<T: Float> core::ops::MulAssign<T> for Complex<T> {
    /// ```rust
    /// use seigyo::Complex;
    ///
    /// let mut a = Complex::from((1., 2.));
    /// 
    /// a *= 2.;
    ///
    /// assert_eq!(a, Complex::from((2., 4.)))
    /// ```
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        self.0 *= rhs;
        self.1 *= rhs;
    }
}

/// ## Add
impl<T: Float> core::ops::Add for Complex<T> {
    type Output = Self;
    
    /// ```rust
    /// use seigyo::Complex;
    ///
    /// let a = Complex::from((1., 2.));
    /// let b = Complex::from((3., 4.));
    /// 
    /// assert_eq!(a + b, Complex::from((4., 6.)))
    /// ```
    fn add(mut self, other: Complex<T>) -> Self::Output {
        self += other;
        self
    }  
}

/// ## Add assign
impl<T: Float> core::ops::AddAssign for Complex<T> {
    /// ```rust
    /// use seigyo::Complex;
    ///
    /// let mut a = Complex::from((1., 2.));
    /// let b = Complex::from((3., 4.));
    ///
    /// a += b;
    ///
    /// assert_eq!(a, Complex::from((4., 6.)))
    /// ```
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
        self.1 += rhs.1;
    }
}

/// Add assign of a floating number
impl<T: Float> core::ops::Add<T> for Complex<T> {
    type Output = Self;

    /// Add a real number to a [`Complex`] number.
    ///
    /// The result is a [`Complex`] number.
    /// ```rust
    /// use seigyo::Complex;
    ///
    /// let a = Complex::new(-8., -10.);
    ///
    /// assert_eq!(a + 91., Complex::new(83., -10.))
    /// ```
    fn add(mut self, rhs: T) -> Self::Output {
        self.0 += rhs;
        self
    }
}

/// Add assign of a floating number
impl<T: Float> core::ops::AddAssign<T> for Complex<T> {
    /// Add a real number to a [`Complex`] number.
    ///
    /// The result is a [`Complex`] number.
    /// ```rust
    /// use seigyo::Complex;
    ///
    /// let mut a = Complex::new(-8., -10.);
    /// a += 91.;
    ///
    /// assert_eq!(a, Complex::new(83., -10.))
    /// ```
    fn add_assign(&mut self, rhs: T) {
        self.0 += rhs;
    }
}

/// ## Iterator sum
impl<T: Float> core::iter::Sum for Complex<T> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), |a, b| a + b)
    }
}

/// ## Subtraction
impl<T: Float> core::ops::Sub for Complex<T> {
    type Output = Self;
    
    /// Subtraction of complex numbers.
    ///
    /// ```rust
    /// use seigyo::Complex;
    ///
    /// let a = Complex::from((1., 2.));
    /// let b = Complex::from((3., 4.));
    /// 
    /// assert_eq!(a - b, Complex::from((-2., -2.)))
    /// ```
    fn sub(mut self, other: Complex<T>) -> Self::Output {
        self -= other;
        self
    }  
}

/// ## Subtraction assign
impl<T: Float> core::ops::SubAssign for Complex<T> {
    /// Subtract assign of two given complex numbers.
    /// ```rust
    /// use seigyo::Complex;
    ///
    /// let mut a = Complex::from((1., 2.));
    /// let b = Complex::from((3., 4.));
    /// 
    /// a -= b;
    /// assert_eq!(a, Complex::from((-2., -2.)))
    /// ```
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
        self.1 -= rhs.1;
    }
}

/// ## Subtraction of a floating number
impl<T: Float> core::ops::Sub<T> for Complex<T> {
    type Output = Self;

    /// Subtract a real number from a [`Complex`] number.
    ///
    /// The result is a [`Complex`] number.
    /// ```rust
    /// use seigyo::Complex;
    ///
    /// let a = Complex::new(22., -10.);
    ///
    /// assert_eq!(a - 12., Complex::new(10., -10.))
    /// ```
    fn sub(mut self, rhs: T) -> Self::Output {
       self.0 -= rhs;
       self
    }
}

/// ## Subtraction of a floating number
impl<T: Float> core::ops::SubAssign<T> for Complex<T> {
    /// Subtract a real number from a [`Complex`] number.
    ///
    /// The result is a [`Complex`] number.
    /// ```rust
    /// use seigyo::Complex;
    ///
    /// let mut a = Complex::new(22., -10.);
    /// a -= 12.;
    ///
    /// assert_eq!(a, Complex::new(10., -10.))
    /// ```
    fn sub_assign(&mut self, rhs: T) {
        self.0 -= rhs;
    }
}

/// ### Negate
/// Negates the complex number
impl<T: Float> core::ops::Neg for Complex<T> {
    type Output = Self;

    /// Negate a complex number.
    /// ```rust
    /// use seigyo::Complex;
    ///
    /// let complex = Complex::from((1., -2.));
    /// assert_eq!(-complex, Complex::from((-1., 2.)));
    /// ```
    #[inline]
    fn neg(self) -> Self::Output {
        Self(-self.0, -self.1)
    }
}

/// ### From
impl<T: Float> From<T> for Complex<T> {
    /// Create a [Complex] number from a floating type.
    /// ```rust
    /// use seigyo::Complex;
    /// 
    /// let complex = Complex::from(-1.);
    /// assert_eq!(complex, Complex::new(-1., 0.));
    /// ```
    fn from(var: T) -> Self {
        Self(var, T::default())
    }
}

impl<T: Float> From<(T, T)> for Complex<T> {
    /// Create a [Complex] number from a tuple of 2 elements of floating type. The first element represents real part and the second part represents the imaginary part.
    /// ```rust
    /// use seigyo::Complex;
    /// 
    /// let complex = Complex::from((3., 4.));
    /// ```
    fn from(var: (T, T)) -> Self {
        Self(var.0, var.1)
    }
}


/// ### Display
impl<T: Float> core::fmt::Display for Complex<T> {
    /// ```rust
    /// use seigyo::Complex;
    ///
    /// // zero
    /// let mut complex = Complex::from(0.);
    /// assert_eq!(complex.to_string(), "0".to_string());
    /// 
    /// // real number
    /// complex = Complex::from(1.0);
    /// assert_eq!(complex.to_string(), "1".to_string());
    /// 
    /// // real fractional number
    /// complex = Complex::from(1.33);
    /// assert_eq!(complex.to_string(), "1.33".to_string());
    /// 
    /// // imaginary number
    /// complex = Complex::new(0., 20.);
    /// assert_eq!(complex.to_string(), "20j".to_string());
    /// 
    /// // imaginary fractional number
    /// complex = Complex::new(0., 20.55);
    /// assert_eq!(complex.to_string(), "20.55j".to_string());
    ///      
    /// complex = Complex::from((1.33, 20.55));
    /// assert_eq!(complex.to_string(), "1.33 + 20.55j".to_string());
    /// ```
   fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> Result<(), core::fmt::Error> {
        let zero = T::default();

        if self.0 == zero {
            write!(f, "{}", if self.1 == zero { "0".to_string() } else { format!("{}j", self.1) })?;
        }
        else {
            write!(f, "{}", self.0)?;
            if self.1 != zero {
                write!(f, " + {}j", self.1)?;
            }
        }
        
        Ok(())
   }
}

impl<T: Float> PartialEq for Complex<T> {
    /// PartialEq for complex number
    fn eq(&self, other: &Self) -> bool { 
        self.0 == other.0 && self.1 == other.1
    }
}

impl<T: Float> PartialEq<T> for Complex<T> {
    /// Compares a complex number to a floating real number which implements [`Float`]
    /// ```rust, should_panic
    /// use seigyo::Complex;
    ///
    /// let complex = Complex::new(16., 1.);
    /// assert_eq!(complex, 1.);
    /// ```
    ///
    /// ```rust
    /// use seigyo::Complex;
    ///
    /// let complex = Complex::from(16.);
    /// assert_eq!(complex, 16.);
    /// ```
    fn eq(&self, other: &T) -> bool {
        self.is_real() && self.0.eq(other)
    }
}

impl PartialEq<Complex<f32>> for f32 {
    /// ```rust
    /// use seigyo::Complex;
    ///
    /// assert_eq!(1.25f32, Complex::new(1.25, 0.));
    /// ```
    ///
    /// ```rust, should_panic
    /// use seigyo::Complex;
    ///
    /// assert_eq!(1.25f32, Complex::new(1.25, 0.01));
    /// ```
    fn eq(&self, other: &Complex<f32>) -> bool {
        other.is_real() && self.eq(&other.0)
    }
}

impl PartialEq<Complex<f64>> for f64 {
    /// ```rust
    /// use seigyo::Complex;
    ///
    /// assert_eq!(1.25f64, Complex::new(1.25, 0.));
    /// ```
    /// 
    /// ```rust, should_panic
    /// use seigyo::Complex;
    ///
    /// assert_eq!(1.25f64, Complex::new(1.25, 0.01));
    /// ```
    fn eq(&self, other: &Complex) -> bool {
        other.is_real() && self.eq(&other.0)
    }
}

impl<T: Float> core::ops::Add<Complex<T>> for f32 {
    type Output = Complex<T>;

    /// ```rust
    /// use seigyo::Complex;
    ///
    /// let c = Complex::new(10.2, -0.33);
    ///
    /// assert_eq!(2f32 + c, Complex::new(12.2, -0.33))
    /// ```
    fn add(self, mut rhs: Complex<T>) -> Self::Output {
        rhs.0 += T::from_f32(self);
        rhs
    }
}

impl<T: Float> core::ops::Add<Complex<T>> for f64 {
    type Output = Complex<T>;

    /// ```rust
    /// use seigyo::Complex;
    ///
    /// let c = Complex::new(10.2, -0.33);
    ///
    /// assert_eq!(2f64 + c, Complex::new(12.2, -0.33))
    /// ```
    fn add(self, mut rhs: Complex<T>) -> Self::Output {
        rhs.0 += T::from_f64(self);
        rhs
    }
}

impl<T: Float> core::ops::Sub<Complex<T>> for f32 {
    type Output = Complex<T>;

    /// ```rust
    /// use seigyo::Complex;
    ///
    /// let c = Complex::new(-29.26, 0.77);
    ///
    /// assert_eq!(3f32 - c, Complex::new(-26.26, 0.77));
    /// ```
    fn sub(self, mut rhs: Complex<T>) -> Self::Output {
        rhs.0 += T::from_f32(self);
        rhs
    }
}

impl<T: Float> core::ops::Sub<Complex<T>> for f64 {
    type Output = Complex<T>;

    /// ```rust
    /// use seigyo::Complex;
    ///
    /// let c = Complex::new(-29.26, 0.77);
    ///
    /// assert_eq!(3f64 - c, Complex::new(-26.26, 0.77));
    /// ```
    fn sub(self, mut rhs: Complex<T>) -> Self::Output {
        rhs.0 += T::from_f64(self);
        rhs
    }
}

/// Defines a matrix
#[derive(Debug, PartialEq, Clone)]
pub struct Matrix<const R: usize, const C: usize, T: Float = f64>([[Complex<T>; C]; R]);

impl<T: Float, const R: usize, const C: usize> core::ops::Deref for Matrix<R, C, T> {
    type Target = [[Complex<T>; C]; R];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl <T: Float, const R: usize, const C: usize> core::ops::DerefMut for Matrix<R, C, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Create a new Matrix by taking ownership of the 2 dimensional array
impl<T: Float, Z: Into<Complex<T>>, const R: usize, const C: usize> From<[[Z; C]; R]> for Matrix<R, C, T> {
    fn from(value: [[Z; C]; R]) -> Self {
        assert_ne!((R, C), (0, 0), "cannot create a matrix with no dimensions");
        Self(value.map(|r| r.map(|c| c.into())))
    }
}

/// Create a new Matrix from a 2 dimensional slice
impl<T: Float, Z: Into<Complex<T>> + Clone, const R: usize, const C: usize> From<&[&[Z; C]; R]> for Matrix<R, C, T> {
    fn from(value: &[&[Z; C]; R]) -> Self {
        assert_ne!((R, C), (0, 0), "cannot create a matrix with no dimension");
        Self(core::array::from_fn(|r| core::array::from_fn(|c| value[r][c].to_owned().into())))
    }
}

/// ## Matrix display
impl<T: Float, const R: usize, const C: usize> core::fmt::Display for Matrix<R, C, T> {
    /// ### Example
    ///
    /// Consider the matrix:$$\begin{bmatrix} 10 & 0 & 20 \\\ 0 & 30 & 0 \\\ 200 & 0 & 100 \end{bmatrix}$$
    /// ```rust
    /// use seigyo::Matrix;
    ///
    /// let matrix = Matrix::from([[10., 0., 20.], [0., 30., 0.], [200., 0., 100.]]);
    /// assert_eq!("│  10  0  20 │\n│   0 30   0 │\n│ 200  0 100 │\n", matrix.to_string());
    /// ```
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // caution: format macro creates heap allocated memory
        let strings: [[String; C]; R] = core::array::from_fn(|i| core::array::from_fn(|j| format!("{}", self.0[i][j])));
        let col_widths: [usize; C] = core::array::from_fn(|j| (0..R).map(|i| strings[i][j].len()).max().unwrap_or(0));

        for s in strings.iter().take(R) {
            write!(f, "│")?;
            for j in 0..C {
                write!(f, " {:>width$}", s[j], width = col_widths[j])?;
            }
            writeln!(f, " │")?;
        }

        Ok(())
    }
}

/// ## Divide
/// divide each matrix entry by the given RHS value.
impl<T: Float, Z: Into<Complex<T>>, const R: usize, const C: usize> core::ops::Div<Z> for Matrix<R, C, T> {
    type Output = Self;

    fn div(mut self, rhs: Z) -> Self::Output {
        self /= rhs;
        self
    }
}

/// ## Divide assign
/// this divide assigns each entry by the given RHS value.
impl<T: Float, Z: Into<Complex<T>>, const R: usize, const C: usize> core::ops::DivAssign<Z> for Matrix<R, C, T> {
    fn div_assign(&mut self, rhs: Z) {
        let denominator = rhs.into();
        self.0.iter_mut().for_each(|r| r.iter_mut().for_each(|c| *c /= denominator));
    }
}

/// ## Multiplication
/// Owned LHS and RHS
impl<T: Float, const R: usize, const C: usize, const O: usize> core::ops::Mul<Matrix<C, O, T>> for Matrix<R, C, T> {
    type Output = Matrix<R, O, T>;

    /// ### Example
    ///
    /// ```rust
    /// use seigyo::Matrix;
    ///
    /// let a = Matrix::from([[4., 1., -4., 5.], [-2., 0., 6., 3.], [2., 7., 8., 9.], [10., -1., -3., 11.]]);
    /// let b = Matrix::from([[4., 1.], [-2., 0.], [2., 7.], [10., 12.]]);
    ///
    /// assert_eq!(a * b, Matrix::from([[56., 36.], [34., 76.], [100., 166.], [146., 121.]]));
    /// ```
    fn mul(self, rhs: Matrix<C, O, T>) -> Self::Output {
        self * &rhs
    }
}

/// Owned LHS and referenced RHS
impl<T: Float, const R: usize, const C: usize, const O: usize> core::ops::Mul<&Matrix<C, O, T>> for Matrix<R, C, T> {
    type Output = Matrix<R, O, T>;

    /// ### Example
    ///
    /// ```rust
    /// use seigyo::Matrix;
    ///
    /// let a = Matrix::from([[4., 1., -4., 5.], [-2., 0., 6., 3.], [2., 7., 8., 9.], [10., -1., -3., 11.]]);
    /// let b = Matrix::from([[4., 1.], [-2., 0.], [2., 7.], [10., 12.]]);
    ///
    /// assert_eq!(a * &b, Matrix::from([[56., 36.], [34., 76.], [100., 166.], [146., 121.]]));
    /// ```
    fn mul(self, rhs: &Matrix<C, O, T>) -> Self::Output {
        Matrix(core::array::from_fn(|i| // self row
            core::array::from_fn(|j| // rhs column
                (0..C).map(|k| self.0[i][k] * rhs.0[k][j]).sum::<Complex<T>>()
            )
        ))
    }
}

/// Referenced LHS and owned RHS
impl<T: Float, const R: usize, const C: usize, const O: usize> core::ops::Mul<Matrix<C, O, T>> for &Matrix<R, C, T> {
    type Output = Matrix<R, O, T>;

    /// ### Example
    ///
    /// ```rust
    /// use seigyo::Matrix;
    ///
    /// let a = Matrix::from([[4., 1., -4., 5.], [-2., 0., 6., 3.], [2., 7., 8., 9.], [10., -1., -3., 11.]]);
    /// let b = Matrix::from([[4., 1.], [-2., 0.], [2., 7.], [10., 12.]]);
    ///
    /// assert_eq!(&a * b, Matrix::from([[56., 36.], [34., 76.], [100., 166.], [146., 121.]]));
    /// ```
    fn mul(self, rhs: Matrix<C, O, T>) -> Self::Output {
        Matrix(core::array::from_fn(|i| // self row
            core::array::from_fn(|j| // rhs column
                (0..C).map(|k| self.0[i][k] * rhs.0[k][j]).sum::<Complex<T>>()
            )
        ))
    }
}

/// Referenced LHS and referenced RHS
impl<T: Float, const R: usize, const C: usize, const O: usize> core::ops::Mul<&Matrix<C, O, T>> for &Matrix<R, C, T> {
    type Output = Matrix<R, O, T>;

    /// Returns a new matrix.
    /// ### Example
    ///
    /// ```rust
    /// use seigyo::Matrix;
    ///
    /// let a = Matrix::from([[4., 1., -4., 5.], [-2., 0., 6., 3.], [2., 7., 8., 9.], [10., -1., -3., 11.]]);
    /// let b = Matrix::from([[4., 1.], [-2., 0.], [2., 7.], [10., 12.]]);
    ///
    /// assert_eq!(&a * &b, Matrix::from([[56., 36.], [34., 76.], [100., 166.], [146., 121.]]));
    /// ```
    fn mul(self, rhs: &Matrix<C, O, T>) -> Self::Output {
        let matrix = core::array::from_fn(|i| // self row
            core::array::from_fn(|j| // rhs column
                (0..C).map(|k| self.0[i][k] * rhs.0[k][j]).sum::<Complex<T>>()
            )
        );

        Matrix(matrix)
    }
}

/// Multiply by a complex
impl<T: Float, Z: Into<Complex<T>>, const R: usize, const C: usize> core::ops::Mul<Z> for Matrix<R, C, T> {
    type Output = Self;

    /// ### Example
    ///
    /// ```rust
    /// use seigyo::Matrix;
    ///
    /// let a = Matrix::from([[1., 8., 3.], [9., 4., 5.], [6., 2., 7.]]);
    /// assert_eq!(a * 2., Matrix::from([[2., 16., 6.], [18., 8., 10.], [12., 4., 14.]]));
    /// ```
    fn mul(mut self, rhs: Z) -> Self::Output {
        self *= rhs;
        self
    }
}

/// ## Multiplication assign
/// For square matrices
impl<T: Float, const C: usize> core::ops::MulAssign for Matrix<C, C, T> {
    /// ### Example
    /// 
    /// ```rust
    /// use seigyo::Matrix;
    ///
    /// let a = Matrix::from([[1., 8., 3.], [9., 4., 5.], [6., 2., 7.]]);
    /// let b = Matrix::from([[6., 7., 4.], [1., 3., 2.], [5., 9., 8.]]);
    /// 
    /// assert_eq!(a * b, Matrix::from([[29., 58., 44.], [83., 120., 84.], [73., 111., 84.]]));
    /// ```
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        let matrix = core::array::from_fn(|i| // row in self
            core::array::from_fn(|j| // col in rhs
                (0..C).map(|k| self.0[i][k] * rhs.0[k][j]).sum::<Complex<T>>()
            )
        );

        *self = Self(matrix);
    }
}

/// Multiply assign by complex number
impl<T: Float, Z: Into<Complex<T>>, const R: usize, const C: usize> core::ops::MulAssign<Z> for Matrix<R, C, T> {
    /// ### Example
    ///
    /// ```rust
    /// use seigyo::Matrix;
    ///
    /// let mut a = Matrix::from([[1., 8., 3.], [9., 4., 5.], [6., 2., 7.]]);
    /// a *= 2.;
    /// assert_eq!(a, Matrix::from([[2., 16., 6.], [18., 8., 10.], [12., 4., 14.]]));
    /// ```
    fn mul_assign(&mut self, rhs: Z) {
        let complex = rhs.into();
        self.0.iter_mut().for_each(|r| r.iter_mut().for_each(|c| *c *= complex));
    }
}

/// ## Addition
impl<T: Float, const R: usize, const C: usize> core::ops::Add for Matrix<R, C, T> {
    type Output = Self;

    /// ### Example
    ///
    /// ```rust
    /// use seigyo::Matrix;
    ///
    /// let a = Matrix::from([[4., 3., 8.], [6., 2., 5.], [1., 5., 9.]]);
    /// let b = Matrix::from([[-7., 13., 1.], [-49., 28., 28.], [28., -17., -10.]]);
    ///
    /// assert_eq!(a + b, Matrix::from([[-3., 16., 9.], [-43., 30., 33.], [29., -12., -1.]]));
    /// ```
    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

/// ## Addition assign
impl<T: Float, const R: usize, const C: usize> core::ops::AddAssign for Matrix<R, C, T> {
    /// ### Example
    ///
    /// ```rust
    /// use seigyo::Matrix;
    ///
    /// let mut a = Matrix::from([[4., 3., 8.], [6., 2., 5.], [1., 5., 9.]]);
    /// let b = Matrix::from([[-7., 13., 1.], [-49., 28., 28.], [28., -17., -10.]]);
    ///
    /// a += b;
    ///
    /// assert_eq!(a, Matrix::from([[-3., 16., 9.], [-43., 30., 33.], [29., -12., -1.]]));
    /// ```
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0.iter_mut().zip(rhs.0).for_each(|(s_arr, o_arr)| s_arr.iter_mut().zip(o_arr).for_each(|(s, o)| *s += o));
    }
}

/// ## Subtraction
impl<T: Float, const R: usize, const C: usize> core::ops::Sub for Matrix<R, C, T> {
    type Output = Self;

    /// ### Example
    ///
    /// ```rust
    /// use seigyo::Matrix;
    ///
    /// let a = Matrix::from([[4., 3., 8.], [6., 2., 5.], [1., 5., 9.]]);
    /// let b = Matrix::from([[-7., 13., 1.], [-49., 28., 28.], [28., -17., -10.]]);
    ///
    /// assert_eq!(a - b, Matrix::from([[11., -10., 7.], [55., -26., -23.], [-27., 22., 19.]]));
    /// ```
    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= rhs;
        self
    }
}

/// ## Subtraction assign
impl<T: Float, const R: usize, const C: usize> core::ops::SubAssign for Matrix<R, C, T> {
    /// ### Example
    ///
    /// ```rust
    /// use seigyo::Matrix;
    ///
    /// let mut a = Matrix::from([[4., 3., 8.], [6., 2., 5.], [1., 5., 9.]]);
    /// let b = Matrix::from([[-7., 13., 1.], [-49., 28., 28.], [28., -17., -10.]]);
    ///
    /// a -= b;
    /// assert_eq!(a, Matrix::from([[11., -10., 7.], [55., -26., -23.], [-27., 22., 19.]]));
    /// ```
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0.iter_mut().zip(rhs.0).for_each(|(s_arr, o_arr)| s_arr.iter_mut().zip(o_arr).for_each(|(s, o)| *s -= o));
    }
}

/// ## Negation
impl<T: Float, const R: usize, const C: usize> core::ops::Neg for Matrix<R, C, T> {
    type Output = Self;

    /// ### Example
    ///
    /// ```rust
    /// use seigyo::Matrix;
    ///
    /// let a = Matrix::from([[4., 3., 8.], [6., 2., 5.], [1., 5., 9.]]);
    /// assert_eq!(-a, Matrix::from([[-4., -3., -8.], [-6., -2., -5.], [-1., -5., -9.]]));
    /// ```
    fn neg(self) -> Self::Output {
        Self(self.0.map(|r| r.map(Complex::neg)))
    }
}

impl<T: Float, const R: usize, const C: usize> Matrix<R, C, T> {
    /// Creates a new matrix with zero entries
    /// ### Example
    /// ```rust
    /// use seigyo::Matrix;
    ///
    /// let matrix = Matrix::new_zero();
    /// assert_eq!(Matrix::from([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]), matrix);
    /// ```
    pub fn new_zero() -> Self {
        assert_ne!((R, C), (0, 0), "Cannot create a matrix with no dimensions");
        Self([[Complex::default(); C]; R])
    }

    /// Returns a new transposed matrix.
    ///
    /// ### Example
    /// ```rust
    /// use seigyo::Matrix;
    ///
    /// let a = Matrix::from([[(4., 0.55), (4., -20.1901), (-4., 3.696969)], [(4., -1.4000006), (8., 15.919191), (8., 22.2)], [(0., 40.00001), (12., -92.99999), (16., 8.867193)]]);
    /// let a_c_t = Matrix::from([[(4., 0.55), (4., -1.4000006), (0., 40.00001)], [(4., -20.1901), (8., 15.919191), (12., -92.99999)], [(-4., 3.696969), (8., 22.2), (16., 8.867193)]]);
    ///
    /// assert_eq!(a.transpose(), a_c_t);
    /// ```
    pub fn transpose(self) -> Matrix<C, R, T> {
        Matrix(core::array::from_fn(|j| core::array::from_fn(|i| self[i][j])))
    }

    /// returns true if the elements of the matrix are **zero** (within tolerance)
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.0.iter().flatten().all(Complex::is_zero)
    }

    /// Returns a new conjugate transposed matrix.
    /// Also known as **Hermitian transpose**.
    ///
    /// ### Example
    /// ```rust
    /// use seigyo::Matrix;
    ///
    /// let a = Matrix::from([[(4., 0.55), (4., -20.1901), (-4., 3.696969)], [(4., -1.4000006), (8., 15.919191), (8., 22.2)], [(0., 40.00001), (12., -92.99999), (16., 8.867193)]]);
    /// let a_c_t = Matrix::from([[(4., -0.55), (4., 1.4000006), (0., -40.00001)], [(4., 20.1901), (8., -15.919191), (12., 92.99999)], [(-4., -3.696969), (8., -22.2), (16., -8.867193)]]);
    ///
    /// assert_eq!(a.conjugate_transpose(), a_c_t);
    /// ```
    pub fn conjugate_transpose(&self) -> Matrix<C, R, T> {
        Matrix(core::array::from_fn(|j| core::array::from_fn(|i| self[i][j].conjugate())))        
    }

    /// ## Rank
    /// Calculates the **RANK** of the given matrix. Rank represents the number of independent columns/rows.
    /// ### Complexity
    /// $$O(R \times C \times \text{rank})$$
    ///
    /// ### Example
    /// ```rust
    /// use seigyo::Matrix;
    ///
    /// let a = Matrix::from([[10., 20., 10.], [-20., -30., 10.], [30., 50., 0.]]);
    ///
    /// assert_eq!(a.rank(), 2);
    /// ```
    pub fn rank(&self) -> usize {
        let (mut matrix, mut rank) = (self.0, C);

        let mut i = 0;
        while i < rank {
            if !matrix[i][i].is_zero() {
                (0..R).filter(|r| *r != i).for_each(|r| {
                    let multiplier = matrix[r][i] / matrix[i][i];
                    (0..rank).for_each(|c| matrix[r][c] -= multiplier * matrix[i][c]);
                });

                // increase count
                i += 1;
            }
            else {
                // find non-zero row
                match (i + 1..C).find(|&r| !matrix[r][i].is_zero()) {
                    Some(r) => (0..rank).for_each(|c| (matrix[r][c], matrix[i][c]) = (matrix[i][c], matrix[r][c])),
                    None => {
                        // reduce rank
                        rank -= 1;
                        // copy the last column (rank) here
                        (0..R).for_each(|r| matrix[r][i] = matrix[r][rank]);
                    },
                }
            }
        }
        
        rank
    }

    /// rounds off the values of the matrix
    /// this may be useful for changing precision of elements of a [Matrix].
    /// For example:
    /// 1. Consider a matrix $M$.
    /// 2. Multiply the matrix by precision value $P$ say 1000.
    /// 3. Round the elements of the matrix: `M.round()`.
    /// 4. Divide the matrix by precision value $P$.
    pub fn round(mut self, decimal_places: u32) -> Self {
        let factor = T::from_f32(10.).powi(decimal_places as i32);
        self.0.iter_mut().for_each(|r| r.iter_mut().for_each(|c| *c = (*c * factor).round() / factor));
        self
    }

    /// ## QR Decomposition
    /// - Decomposes the given matrix into two matrices using QR decomposition.
    /// - The first matrix Q is a **Unitary matrix**. Meaning, it's conjugate transpose is equal to it's inverse.$$Q^{\dagger} = Q^{-1}$$
    /// - The QR decomposition performed here uses Householder reflections because it is more numerically stable than gaussian eliminations.
    /// - Takes the ownership of the matrix decomposing it into corresponding Q and R matrices.
    /// 
    /// Let there be a matrix $A_{R \times C}$.
    /// $$A_{R \times C} = Q_{R \times C} R_{C \times C}$$
    /// Error is returned if any of the columns are dependent.
    ///
    /// **NOTE:** The matrix Q represents the Gram Schmidt matrix.
    ///
    /// ### Uses
    /// - Solving least squares problem
    /// - Eigen value computation
    ///
    /// ### Example
    /// ```rust
    /// use seigyo::Matrix;
    ///
    /// let a = Matrix::from([[12., -51., 4.], [6., 167., -68.], [-4., 24., -41.]]);
    /// let (q, r) = {
    ///     let (q, r) = a.qr().unwrap();
    ///     // round to 2 decimal places
    ///     (q.round(2), r.round(2))
    /// };
    ///
    /// assert_eq!(q, Matrix::from([
    ///     [-0.86, 0.39, 0.33],
    ///     [-0.43, -0.9, -0.03],
    ///     [0.29, -0.17, 0.94]
    /// ]));
    /// 
    /// assert_eq!(r, Matrix::from([
    ///     [-14., -21., 14.],
    ///     [0., -175., 70.],
    ///     [0., 0., -35.]
    /// ]));
    /// ```
    ///
    /// ```rust, should_panic
    /// use seigyo::Matrix;
    ///
    /// // Dependent columns
    /// let a = Matrix::from([[3., 3.], [1., 1.]]);
    /// a.qr().unwrap();
    /// ```
    pub fn qr(self) -> Result<(Matrix<R, R, T>, Matrix<R, C, T>), Error> {
        // todo: for m < n
        let mut q = Matrix::new_identity();
        let mut r = self;

        let two = T::from_f32(2.);

        for i in 0..C {
            // norm of i-th column (rows i..R)
            let norm_x_sq = (i..R).map(|k| r[k][i].normsq()).sum::<T>();
            let norm = norm_x_sq.sqrt();
            if norm < T::tolerance() {
                return Err(Error::SingularMatrix);
            }

            let alpha = match r[i][i].is_zero() {
                true => norm.into(),
                false => -(r[i][i].normalize() * norm),
            };

            let v_norm_sq = two * (norm_x_sq - r[i][i].dot(alpha));
            if v_norm_sq.sqrt() < T::tolerance() {
                continue;
            }
                
            let tau = two / v_norm_sq;

            // unnormalized Householder vector v[i..R], zero-padded for k < i
            let v = core::array::from_fn::<Complex<T>, R, _>(|k| match k.cmp(&i) {
                core::cmp::Ordering::Less => Complex::default(),
                core::cmp::Ordering::Equal => r[k][i] - alpha,
                core::cmp::Ordering::Greater => r[k][i],
            });

            // column-by-column: dot is fully computed before its column is written
            for j in i..C {
                let dot = (i..R).map(|k| v[k].conjugate() * r[k][j]).sum::<Complex<T>>();
                (i..R).for_each(|k| r[k][j] -= v[k] * dot * tau);
            }

            // row-by-row: dot is fully computed before its row is written
            for j in 0..R {
                let dot = (i..R).map(|k| q[j][k] * v[k]).sum::<Complex<T>>();
                (i..R).for_each(|k| q[j][k] -= dot * v[k].conjugate() * tau);
            }
        }
        
        Ok((q, r))
    }

    /// ## SVD
    /// Deomposes the matrix into three matrices.
    ///
    /// ### Example
    /// ```rust
    /// use seigyo::Matrix;
    ///
    /// let a = Matrix::from([
    ///     [0.8595, 0.7176, 0.9781, 0.8116, 0.2164],
    ///     [0.8886, 0.05399, 0.2921, 0.7058, 0.1801],
    ///     [0.8149, 0.367, 0.04329, 0.5527, 0.7479],
    ///     [0.7431, 0.9701, 0.9428, 0.5411, 0.0009715],
    ///     [0.8033, 0.8404, 0.9647, 0.9118, 0.8811],
    ///     [0.05875, 0.4113, 0.03543, 0.1149, 0.8648],
    ///     [0.7246, 0.3075, 0.4898, 0.8406, 0.5857],
    ///     [0.538, 0.5798, 0.4514, 0.6041, 0.01276],
    ///     [0.7342, 0.001529, 0.2108, 0.426, 0.5745],
    ///     [0.6983, 0.7891, 0.4445, 0.2376, 0.1985],
    /// ]);
    /// println!("original:\n{a}");
    ///
    /// let (u, m, v) = {
    ///     let (u, m, v) = a.svd().unwrap();
    ///     let v_t = v.clone().conjugate_transpose();
    ///     assert!(u.is_orthogonal(), "u not orthogonal");
    ///     assert!(v.is_orthogonal(), "v not orthogonal");
    /// 
    ///     println!("product:\n{}", &u * &m * v_t);
    ///     (u.round(2), m.round(2), v.round(2))
    /// };
    ///
    /// todo!("not yet implemented")
    /// ```
    pub fn svd(self) -> Result<(Matrix<R, R, T>, Matrix<R, C, T>, Matrix<C, C, T>), Error> {
        let mut u = Matrix::new_identity();
        let mut m = self;
        let mut v = Matrix::new_identity();

        let two = T::from_f32(2.);

        for i in 0..C {
            // column reflection block
            {
                // norm of i-th column (rows i..R)
                let norm_x_sq = (i..R).map(|k| m[k][i].normsq()).sum::<T>();
                let norm = norm_x_sq.sqrt();
                if norm < T::tolerance() {
                    return Err(Error::SingularMatrix);
                }

                let alpha = match m[i][i].is_zero() {
                    true => norm.into(),
                    false => -(m[i][i].normalize() * norm),
                };

                let v_norm_sq = two * (norm_x_sq - m[i][i].dot(alpha));

                // if norm is less than tolerence, the column is already reduced
                if v_norm_sq.sqrt() >= T::tolerance() {
                    let tau = two / v_norm_sq;

                    // unnormalized Householder vector v[i..R], zero-padded for k < i
                    let x = core::array::from_fn::<Complex<T>, R, _>(|k| match k.cmp(&i) {
                        core::cmp::Ordering::Less => Complex::default(),
                        core::cmp::Ordering::Equal => m[k][i] - alpha,
                        core::cmp::Ordering::Greater => m[k][i],
                    });

                    // column-by-column: dot is fully computed before its column is written
                    for j in i..C {
                        let dot = (i..R).map(|k| x[k].conjugate() * m[k][j]).sum::<Complex<T>>();
                        (i..R).for_each(|k| m[k][j] -= x[k] * dot * tau);
                    }

                    // row-by-row: dot is fully computed before its row is written
                    for j in 0..R {
                        let dot = (i..R).map(|k| u[j][k] * x[k]).sum::<Complex<T>>();
                        (i..R).for_each(|k| u[j][k] -= dot * x[k].conjugate() * tau);
                    }
                }
            }

            // row reflection block
            // if C <= 2, then it is already row normalized
            if C > 2 && i < C - 2 {
                let pc = i + 1; // pivot column
                let norm_sq = (pc..C).map(|j| m[i][j].normsq()).sum::<T>();
                let norm = norm_sq.sqrt();

                // if norm from pivot is zero in the row, the row is already reduced
                if norm >= T::tolerance() {
                    let alpha = match m[i][pc].is_zero() {
                        true => norm.into(),
                        false => -(m[i][pc].normalize() * norm),
                    };

                    let v_norm_sq = two * (norm_sq - m[i][pc].dot(alpha));

                    // if norm of elements after pivot is zero, the row is already reduced
                    if v_norm_sq.sqrt() >= T::tolerance() {
                        let tau = two / v_norm_sq;

                        let x = core::array::from_fn::<Complex<T>, C, _>(|j| match j.cmp(&pc) {
                            core::cmp::Ordering::Less => Complex::default(),
                            core::cmp::Ordering::Equal => m[i][j] - alpha,
                            core::cmp::Ordering::Greater => m[i][j],
                        });

                        for j in i..R {
                            let dot = (pc..C).map(|k| x[k].conjugate() * m[j][k]).sum::<Complex<T>>();
                            (pc..C).for_each(|k| m[j][k] -= x[k] * dot * tau);
                        }

                        for j in 0..C {
                            let dot = (pc..C).map(|k| v[j][k] * x[k]).sum::<Complex<T>>();
                            (pc..C).for_each(|k| v[j][k] -= dot * x[k].conjugate() * tau);
                        }
                    }
                }
            }
        }

        // apply givens rotation on m
        todo!("givens rotation");

        Ok((u, m, v))
    }

    /// ## Column normalization
    /// Method to normalize the columns of a given matrix.
    /// ### Example
    /// ```rust
    /// use seigyo::Matrix;
    ///
    /// let mut a = Matrix::from([[12., -51., 4.], [6., 167., -68.], [-4., 24., -41.]]);
    /// a.normalize_columns();
    /// let normalized_a = Matrix::from([[0.8571428571428571, -0.2893526786447601, 0.05031148036146422], [0.42857142857142855, 0.9474881830132341, -0.8552951661448918], [-0.2857142857142857, 0.1361659664210636, -0.5156926737050083]]);
    ///
    /// assert_eq!(normalized_a, a);
    /// ```
    pub fn normalize_columns(&mut self) {
        for c in 0..C {
            let sum = (0..R).map(|r| self.0[r][c].normsq()).sum::<T>().sqrt();
            (0..R).for_each(|r| self[r][c] /= sum);
        }
    }

    /// ## Normal columns check
    /// Returns true if the columns of the matrix are normalized.
    /// 
    /// Exact `1` is unrealistic. Tolerance is used to determine how close the sum is to `1`.
    pub fn has_normal_columns(&self) -> bool {
        (0..C).map(|c| // cols
            (0..R).map(|r| self.0[r][c].normsq()).sum::<T>().sqrt() // sum all elements in the row
        ).all(|sum| (sum - T::from_f32(1.)).abs() <= T::tolerance())
    }


    /// Method to determine whether the columns of the matrix are orthogonal or not.
    ///
    /// For tall matrices (R > C), the following condition is checked:$$A^TA = I$$
    /// 
    /// For fat matrices (C > R), the following condition is checked:$$AA^T = I$$
    /// ### Example
    /// 1. Orthogonal matrix
    /// ```rust
    /// use seigyo::Matrix;
    ///
    /// let a = Matrix::from([
    ///     [3., -6., 2.],
    ///     [2., 3., 6.],
    ///     [6., 2., -3.],
    /// ]) / 7.;
    ///
    /// assert!(a.is_orthogonal(), "matrix not orthogonal");
    /// ```
    /// 2. Non-orthogonal matrix (Fat)
    /// ```rust
    /// use seigyo::Matrix;
    ///
    /// let a = Matrix::from([
    ///     [-4.4882, 2.8336, -0.2785, 1.9537, -2.2951],
    ///     [-1.8932, -0.3416, 0.2267, -3.1572, 0.6278],
    ///     [-2.9985, -2.0746, 2.012, -0.6371, 2.4282],
    ///     [-1.8012, -1.7131, -1.0254, -3.9112, 3.3598],
    /// ]);
    ///
    /// assert!(!a.is_orthogonal(), "matrix is orthogonal");
    /// ```
    /// 3. Non-orthogonal matrix (Tall)
    /// ```rust
    /// use seigyo::Matrix;
    ///
    /// let a = Matrix::from([
    ///     [8.0594, -2.5606, -3.094],
    ///     [14.0597, 13.5206, 3.4394],
    ///     [7.8801, 8.0084, 11.4464],
    ///     [6.8671, 13.1027, 11.2753],
    ///     [-1.7049, -3.6269, -3.3591],
    /// ]);
    ///
    /// assert!(!a.is_orthogonal(), "matrix is orthogonal");
    /// ```
    pub fn is_orthogonal(&self) -> bool {
        let transpose = self.to_owned().transpose();

        (C > R && (self * &transpose).is_identity())
            || 
        ((transpose * self).is_identity())
    }

    /// Checks if a matrix represents identity matrix or not.
    ///
    /// The values are checked against the Tolerence specified for the [`Float`] type.
    #[inline]
    fn is_identity(&self) -> bool {
        self.iter().enumerate().all(|(r, row)| row.iter().enumerate().all(|(c, val)| (r == c && val.is_one()) || (r != c && val.is_zero())))
    }
}

impl<T: Float, const C: usize> Matrix<C, C, T> {
    /// Creates a new **identity matrix**.
    /// ```rust
    /// use seigyo::Matrix;
    ///
    /// let matrix = Matrix::new_identity();
    /// assert_eq!(Matrix::from([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]), matrix);
    /// ```
    pub fn new_identity() -> Self {
        assert_ne!(C, 0, "Cannot create a matrix with no dimensions");
        Self(core::array::from_fn(|i| core::array::from_fn(|j| match j == i {
            true => T::from_f32(1.),
            false => T::default(),
        }.into())))
    }

    /// ## Trace
    /// Calculates the sum of the diagonal elements of the matrix
    pub fn trace(&self) -> Complex<T> {
        (0..C).map(|i| self.0[i][i]).sum()
    }

    /// Calculates the determinant of the given square matrix
    ///
    /// ### Procedure
    /// - Uses Gaussian elemination and transformations to reduce the matrix to upper triangular form.
    /// - The determinant is then the product of all diagonal elements.
    ///
    /// ### Complexity
    /// $O(C^3)$
    ///
    /// ### Example
    /// ```rust
    /// use seigyo::{Matrix, Complex};
    ///
    /// let a = Matrix::from([[0., 12., 16.], [4., 4., -4.], [4., 8., 8.]]);
    /// assert_eq!(a.determinant(), Complex::from(-320.))
    /// ```
    pub fn determinant(&self) -> Complex<T> {
        let mut matrix = self.0;
        let [mut det, mut total] = [Complex::from(T::from_f32(1.)); 2];

        for diag_row in 0..C {
            // swap row with non-zero element
            let Some(non_zero_row) = (diag_row..C).find(|&i| !matrix[i][diag_row].is_zero()) else { continue; };

            // if diagonal row is not the non-zero row
            if non_zero_row != diag_row {
                // swap rows
                (matrix[non_zero_row], matrix[diag_row]) = (matrix[diag_row], matrix[non_zero_row]);

                // change determinant sign if odd
                if (non_zero_row - diag_row) & 1 == 1 {
                    det = -det;
                }
            }

            // transform every row below diag_row
            let temp: [Complex<T>; C] = core::array::from_fn(|j| matrix[diag_row][j]);
            for row in matrix.iter_mut().skip(diag_row + 1) {
                let num2 = row[diag_row];
                row.iter_mut().zip(temp).for_each(|(k, ele)| *k = (temp[diag_row] * *k) - (num2 * ele));
                total *= temp[diag_row];
            }
        }

        // multiply diagonal elements
        (0..C).for_each(|i| det *= matrix[i][i]);
        det / total
    }

    /// ## Is Singular
    /// Determines whether the matrix is singular or not
    ///
    /// Uses partial LU decomposition.
    /// 
    /// ### Example
    /// 1. singular matrix
    /// ```rust
    /// use seigyo::Matrix;
    ///
    /// let a = Matrix::from([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
    /// assert!(a.is_singular());
    /// ```
    /// 2. non-singular matrix
    /// ```rust
    /// use seigyo::Matrix;
    ///
    /// let a = Matrix::from([[0., 12., 16.], [4., 4., -4.], [4., 8., 8.]]);
    /// assert!(!a.is_singular());
    /// ```
    pub fn is_singular(&self) -> bool {
        let mut matrix = self.0;

        // iterate over column
        for k in 0..C {
            let (pivot_row, pivot_val_mag) = matrix
                .iter()
                .enumerate()
                .skip(k + 1)
                .map(|(i, row)| (i, row[k].magnitude()))
                .fold((k, matrix[k][k].magnitude()), |max, cur|
                    if cur.1 > max.1 { cur } else { max }
                );

            // return if pivot value is zero
            if pivot_val_mag < T::tolerance() {
                return true;
            }

            // swap rows if required
            if pivot_row != k {
                (matrix[k], matrix[pivot_row]) = (matrix[pivot_row], matrix[k]);
            }

            // elimination below pivot
            (k + 1..C).for_each(|r| {
                // storing multiplier in matrix[r][k]
                matrix[r][k] /= matrix[k][k];
                (k + 1..C).for_each(|c| matrix[r][c] -= matrix[r][k] * matrix[k][c]);
            });
        }

        false
    }

    /// Determines the inverse of a matrix
    /// ### Procedure
    /// Gaussian Jordan Elemination Method
    ///
    /// ### Complexity
    /// $O(C^3)$
    ///
    /// ### Note
    /// Since we are not using rationals, **floating point inacuracy** might be encountered.
    /// This means, the solution might output values slightly varying
    ///
    /// ### Example
    /// 1. Non-singular matrix
    ///
    /// ```rust
    /// use seigyo::Matrix;
    ///
    /// const PRECISION: u32 = 2;
    ///
    /// let a = Matrix::from([[2., -1., 0.], [-1., 2., -1.], [0., -1., 2.]]);
    /// let inverse = a.inverse().unwrap().round(2);
    ///
    /// assert_eq!(inverse, Matrix::from([[0.75, 0.5, 0.25], [0.5, 1., 0.5], [0.25, 0.5, 0.75]]))
    /// ```
    /// 2. Singular matrix
    ///
    /// ```rust, should_panic
    /// use seigyo::Matrix;
    ///
    /// let a = Matrix::from([[1., -2., 2.], [-1., 2., -2.], [3., -2., -1.]]);
    /// a.inverse().unwrap();
    /// ```
    pub fn inverse(&self) -> Result<Self, Error> {
        let mut matrix = self.0;
        let mut inverse = Self::new_identity();

        for i in 0..C {
            let max_row = match (i..C).max_by(|&a, &b| {
                let magnitude = |z: Complex<T>| z.real().abs() + z.imaginary().abs();
                magnitude(matrix[a][i]).partial_cmp(&magnitude(matrix[b][i])).unwrap_or(core::cmp::Ordering::Equal)
            }) {
                Some(max_row) if !matrix[max_row][i].is_zero() => max_row,
                _ => return Err(Error::SingularMatrix),
            };

            // swap row
            if max_row != i {
                (matrix[max_row], matrix[i]) = (matrix[i], matrix[max_row]);
                (inverse[max_row], inverse[i]) = (inverse[i], inverse[max_row]);
            }

            let pivot = matrix[i][i];
            (0..C).for_each(|c| {
                matrix[i][c] /= pivot;
                inverse[i][c] /= pivot;
            });

            (0..C).filter(|&r| r != i).for_each(|r| {
                let factor = matrix[r][i];
                (0..C).for_each(|k| {
                    matrix[r][k] -= factor * matrix[i][k];
                    inverse.0[r][k] -= factor * inverse.0[i][k];
                });
            });
        }

        Ok(inverse)
    }

    /// Returns true if the matrix is symmetric
    /// ```rust
    /// use seigyo::Matrix;
    ///
    /// let m = Matrix::from([
    ///     [0., -12.5, 1.12, 0.333],
    ///     [-12.5, 59.636, -8.92, -901.],
    ///     [1.12, -8.92, 0.118, -59.636],
    ///     [0.333, -901., -59.636, 112.3]
    /// ]);
    ///
    /// assert!(m.is_symmetric());
    /// ```
    pub fn is_symmetric(&self) -> bool {
        (0..C).all(|i| (i + i..C).all(|j| self[i][j] == self[j][i]))
    }
}

impl<T: Float> Matrix<3, 1, T> {
    /// ## Cross product
    /// Returns the cross product of two 3D column vectors.
    ///
    /// For vectors `a` and `b`, computes `a × b`:
    /// ```text
    /// | i   j   k  |
    /// | a0  a1  a2 |
    /// | b0  b1  b2 |
    /// ```
    /// ### Example
    /// ```rust
    /// use seigyo::Matrix;
    ///
    /// let a = Matrix::from([[1.], [0.], [0.]]);
    /// let b = Matrix::from([[0.], [1.], [0.]]);
    /// assert_eq!(a.cross(b), Matrix::from([[0.], [0.], [1.]]));
    /// ```
    pub fn cross(&self, other: Self) -> Self {
        let [a0, a1, a2] = [self[0][0], self[1][0], self[2][0]];
        let [b0, b1, b2] = [other[0][0], other[1][0], other[2][0]];
        Matrix::from([
            [a1 * b2 - a2 * b1],
            [a2 * b0 - a0 * b2],
            [a0 * b1 - a1 * b0],
        ])
    }

    /// ## Skew Symmetric
    /// Method to make skew symmetric of a 3x1 matrix
    fn skew_symmetric(self) -> Matrix<3, 3, T> {
        let zero = Complex::default();
        Matrix::from([
            [zero, -self[2][0], self[1][0]],
            [self[2][0], zero, -self[0][0]],
            [-self[1][0], self[0][0], zero],
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn new_dimensionless_zero_matrix() {
        Matrix::<0, 0, f32>::new_zero();
    }

    #[test]
    #[should_panic]
    fn new_dimensionless_identity_matrix() {
        Matrix::<0, 0, f32>::new_identity();
    }

    #[test]
    #[should_panic]
    fn inverse_of_singular_matrix() {
        let a = Matrix::from([[1., -2., 2.], [-1., 2., -2.], [3., -2., -1.]]);
        a.inverse().unwrap();
    }
}

/// # Transformation
/// Transformation composes of the rotation and translation components.
/// 
/// **Uses:**
/// - To represent configuration of a rigid body with respect to a fixed frame represented by [`Transformation::default()`].
/// - To change the reference frame in which a vector or frame is represented.
/// - To displace a vector or frame.
#[derive(Debug, Clone)]
pub struct Transformation<T: Float> {
    /// the 3x3 rotation matrix
    rotation: Matrix<3, 3, T>,
    /// the 3x1 translation matrix
    translation: Matrix<3, 1, T>,
}

impl<T: Float> Default for Transformation<T> {
    /// - Gives a transformation matrix which performs no displacement.
    /// - It can also be thought of to be representing origin.
    /// - Identity 4x4 matrix.
    ///
    /// ### Example
    /// ```rust
    /// use seigyo::{Transformation, Matrix};
    ///
    /// let default_transformation = Transformation::default();
    /// let matrix = Matrix::from([
    ///     [1., 0., 0., 0.],
    ///     [0., 1., 0., 0.],
    ///     [0., 0., 1., 0.],
    ///     [0., 0., 0., 1.],
    /// ]);
    ///
    /// assert_eq!(default_transformation, matrix);
    /// ```
    fn default() -> Self {
        Self {
            rotation: Matrix::new_identity(),
            translation: Matrix::new_zero(),
        }
    }
}

/// ## Multiply assign
impl<T: Float> core::ops::Mul for Transformation<T> {
    type Output = Self;

    /// Multiplication of transformation matrices is a transformation matrix.
    /// 
    /// ```rust
    /// use seigyo::{Transformation, Matrix};
    ///
    /// // body b with respect to frame s
    /// let sb = Matrix::from([
    ///     [0., 0., 1., 400.],
    ///     [0., 1., 0., 50.],
    ///     [1., 0., 0., 300.],
    ///     [0., 0., 0., 1.],
    /// ]);
    ///
    /// // body c with respect to frame b
    /// let bc = Matrix::from([
    ///     [0., 0., 1., 300.],
    ///     [0., 1., 0., 100.],
    ///     [1., 0., 0., 120.],
    ///     [0., 0., 0., 1.],
    /// ]);
    ///
    /// let sc = Matrix::from([
    ///     [1., 0., 0., 520.],
    ///     [0., 1., 0., 150.],
    ///     [0., 0., 1., 600.],
    ///     [0., 0., 0., 1.],
    /// ]);
    ///
    /// assert_eq!(Transformation::try_from(sb).unwrap() * Transformation::try_from(bc).unwrap(), Transformation::try_from(sc).unwrap());
    /// ```
    fn mul(mut self, rhs: Self) -> Self::Output {
        self *= rhs;
        self
    }
}

/// ## Multiply assign
impl<T: Float> core::ops::MulAssign for Transformation<T> {
    /// Multiplication of transformation matrices is also a transformation matrix
    /// ```rust
    /// use seigyo::{Transformation, Matrix};
    ///
    /// // body b with respect to frame s
    /// let mut sb = Matrix::from([
    ///     [0., 0., 1., 400.],
    ///     [0., 1., 0., 50.],
    ///     [1., 0., 0., 300.],
    ///     [0., 0., 0., 1.],
    /// ]);
    ///
    /// // body c with respect to frame b
    /// let bc = Matrix::from([
    ///     [0., 0., 1., 300.],
    ///     [0., 1., 0., 100.],
    ///     [1., 0., 0., 120.],
    ///     [0., 0., 0., 1.],
    /// ]);
    ///
    /// let sc = Matrix::from([
    ///     [1., 0., 0., 520.],
    ///     [0., 1., 0., 150.],
    ///     [0., 0., 1., 600.],
    ///     [0., 0., 0., 1.],
    /// ]);
    /// 
    /// sb *= bc;
    ///
    /// assert_eq!(Transformation::try_from(sb).unwrap(), Transformation::try_from(sc).unwrap());
    /// ```
    fn mul_assign(&mut self, rhs: Self) {
        self.translation += &self.rotation * rhs.translation;
        self.rotation *= rhs.rotation;
    }
}

impl<T: Float> Transformation<T> {
    /// ### New transformation matrix
    /// Creates a new transformation matrix.
    ///
    /// Returns [`Error`] in case the rotation matrix is invalid.
    pub fn new(rotation: Matrix<3, 3, T>, translation: Matrix<3, 1, T>) -> Result<Self, Error> {
        Self::validate_rotation(&rotation)?;

        Ok(Self {
            rotation,
            translation,
        })
    }

    /// ### Transformation inverse
    /// Method to find inverse of the transformation matrix
    /// $$T^{-1} = \begin{bmatrix} R & p \\ 0 & 1 \end{bmatrix}^{-1} = \begin{bmatrix} R^T & -R^Tp \\ 0 & 1\end{bmatrix}$$
    /// Also,$$R^TR = I$$
    pub fn inverse(self) -> Self {
        let rotation = self.rotation.transpose();
        let translation = -(&rotation * &self.translation);

        Self {
            rotation,
            translation,
        }
    }

    /// Method to validate the rotation matrix.
    /// 
    /// Normally only 3 entries (out of 9) can be selcted independently in a rotation matrix. These correspond to the angles to be rotated by each axis.
    ///
    /// The rotation matrix has 9 elements. Hence 6 constraints need to be applied.
    /// 1) The unit norm condition: $\hat{x}_b$, $\hat{y}_b$, $\hat{z}_b$ are all unit vectors. $$\begin{aligned} r_{11}^2 + r_{21}^2 + r_{31}^2 &= 1 \\ r_{12}^2 + r_{22}^2 + r_{32}^2 &= 1 \\ r_{13}^2 + r_{23}^2 + r_{33}^2 &= 1 \end{aligned}$$
    /// 2) The orthogonality condition (dot product of columns is zero): $$\begin{aligned} \hat{x}_b . \hat{y}_b &= r_{11}r_{12} + r_{21}r_{22} + r_{31}r_{32} &= 0 \\ \hat{y}_b . \hat{z}_b &= r_{12}r_{13} + r_{22}r_{23} + r_{32}r_{32} &= 0 \\ \hat{x}_b . \hat{z}_b &= r_{11}r_{13} + r{21}r_{23} + r_{31}r_{33} &= 0 \end{aligned}$$
    ///
    /// Also for the frame to be right handed, an additional condition should be that the determinant should be `+1`.
    /// If the determinant is `-1`, it would lead to **reflections** or **inversions**.
    ///
    /// A rotaion matrix can be invalid because of the following reasons:
    /// 1. the given matrix is singular [`Error::RotationSingular`].
    /// 2. the given matrix doesn't have orthogonal columns [`Error::RotationNonOrthogonal`].
    /// 3. the given matrix doesn't have a positive determinant ( = 1) [`Error::RotationInvalidOrientation`].
    fn validate_rotation(rotation: &Matrix<3, 3, T>) -> Result<(), TransformationError> {
        // matrix should not be singular
        if rotation.is_singular() {
            return Err(TransformationError::RotationSingular);
        }

        // columns should be orthogonal
        if [(0, 1), (0, 2), (1, 2)].iter().any(|&(c1, c2)| !(0..3).map(|i| rotation[i][c1] * rotation[i][c2]).sum::<Complex<T>>().is_zero()) {
            return Err(TransformationError::RotationNonOrthogonal);
        }

        // check orientation: determinant should be real and equal to 1
        let determinant = rotation.determinant();
        if determinant.is_imaginary() || (determinant.real() - T::from_f32(1.)).abs() > T::tolerance() {
            return Err(TransformationError::RotationInvalidOrientation);
        }

        Ok(())
    }

    /// ### Transformation adjoint
    /// Returns the 6x6 adjoint matrix for the transformation matrix
    /// ```rust
    /// use seigyo::{Matrix, Transformation};
    /// 
    /// let m = Matrix::from([
    ///     [0., 0., 1., 300.],
    ///     [0., 1., 0., 100.],
    ///     [1., 0., 0., 120.],
    ///     [0., 0., 0., 1.],
    /// ]);
    ///
    /// let transformation = Transformation::try_from(m).expect("invalid matrix");
    ///
    /// let adjoint = transformation.to_adjoint();
    ///
    /// assert_eq!(adjoint, Matrix::from([
    ///     [0., 0., 1., 0., 0., 0.],
    ///     [0., 1., 0., 0., 0., 0.],
    ///     [1., 0., 0., 0., 0., 0.],
    ///     [100., -120., 0., 0., 0., 1.],
    ///     [-300., 0., 120., 0., 1., 0.],
    ///     [0., 300., -100., 1., 0., 0.],
    /// ]));
    ///
    /// // adjoints have determinant 1
    /// assert_eq!(adjoint.determinant(), 1., "adjoint should always have determinant one");
    /// ```
    pub fn to_adjoint(self) -> Matrix<6, 6, T> {
        let one_zero = self.translation.skew_symmetric() * &self.rotation;
        let zero = Complex::default();

        Matrix::from([
            [self.rotation[0][0], self.rotation[0][1], self.rotation[0][2], zero, zero, zero],
            [self.rotation[1][0], self.rotation[1][1], self.rotation[1][2], zero, zero, zero],
            [self.rotation[2][0], self.rotation[2][1], self.rotation[2][2], zero, zero, zero],
            [one_zero[0][0], one_zero[0][1], one_zero[0][2], self.rotation[0][0], self.rotation[0][1], self.rotation[0][2]],
            [one_zero[1][0], one_zero[1][1], one_zero[1][2], self.rotation[1][0], self.rotation[1][1], self.rotation[1][2]],
            [one_zero[2][0], one_zero[2][1], one_zero[2][2], self.rotation[2][0], self.rotation[2][1], self.rotation[2][2]],
        ])
    }
}

impl<T: Float> TryFrom<Matrix<3, 3, T>> for Transformation<T> {
    type Error = Error;

    /// New transformation matrix from a rotation matrix
    /// ```rust
    /// use seigyo::{Matrix, Transformation};
    /// 
    /// let m = Matrix::from([
    ///     [0., -1., 0.],
    ///     [0., 0., 1.],
    ///     [1., 0., 0.],
    /// ]);
    ///
    /// let transformation = Transformation::try_from(m).expect("invalid rotation matrix");
    ///
    /// assert_eq!(transformation, Matrix::from([
    ///     [0., -1., 0., 0.],
    ///     [0., 0., 1., 0.],
    ///     [1., 0., 0., 0.],
    ///     [0., 0., 0., 1.],
    /// ]));
    /// ```
    fn try_from(rotation: Matrix<3, 3, T>) -> core::result::Result<Self, Self::Error> {
        Self::validate_rotation(&rotation)?;

        Ok(Self {
            rotation,
            ..Default::default()
        })
    }
}

impl<T: Float> TryFrom<Matrix<4, 4, T>> for Transformation<T> {
    type Error = Error;

    /// ```rust
    /// use seigyo::{Matrix, Transformation};
    /// 
    /// let m = Matrix::from([
    ///     [0., -1., 0., -20.23],
    ///     [0., 0., 1., 0.55],
    ///     [1., 0., 0., 2.],
    ///     [0., 0., 0., 1.],
    /// ]);
    ///
    /// let transformation = Transformation::try_from(m).expect("invalid rotation matrix");
    ///
    /// assert_eq!(transformation, Matrix::from([
    ///     [0., -1., 0., -20.23],
    ///     [0., 0., 1., 0.55],
    ///     [1., 0., 0., 2.],
    ///     [0., 0., 0., 1.],
    /// ]));
    /// ```
    fn try_from(value: Matrix<4, 4, T>) -> core::result::Result<Self, Self::Error> {
        // rotation
        let rotation = Matrix::from([
            [value[0][0], value[0][1], value[0][2]],
            [value[1][0], value[1][1], value[1][2]],
            [value[2][0], value[2][1], value[2][2]],
        ]);

        Self::validate_rotation(&rotation)?;

        let translation = Matrix::from([
            [value[0][3]],
            [value[1][3]],
            [value[2][3]],
        ]);


        Ok(Self {
            rotation,
            translation,
        })
    }
}

impl<T: Float> From<Matrix<3, 1, T>> for Transformation<T> {
    /// Transformation matrix from a translation matrix
    /// ```rust
    /// use seigyo::{Transformation, Matrix};
    ///
    /// let translation = Matrix::from([
    ///     [1.],
    ///     [2.],
    ///     [3.],
    /// ]);
    ///
    /// let matrix = Matrix::from([
    ///     [1., 0., 0., 1.],
    ///     [0., 1., 0., 2.],
    ///     [0., 0., 1., 3.],
    ///     [0., 0., 0., 1.],
    /// ]);
    ///
    /// assert_eq!(Transformation::from(translation), matrix);
    /// ```
    fn from(translation: Matrix<3, 1, T>) -> Self {
        Self {
            translation,
            ..Default::default()
        }
    }
}

impl<T: Float> From<Transformation<T>> for Matrix<4, 4, T> {
    fn from(value: Transformation<T>) -> Self {
        let r = value.rotation;
        let t = value.translation;
        let z = Complex::default();
        let o = T::from_f32(1.).into();

        Self::from([
            [r[0][0], r[0][1], r[0][2], t[0][0]],
            [r[1][0], r[1][1], r[1][2], t[1][0]],
            [r[2][0], r[2][1], r[2][2], t[2][0]],
            [z, z, z, o],
        ])
    }
}

impl<T: Float> PartialEq<Matrix<4, 4, T>> for Transformation<T> {
    /// Compares transformation matrix to a 4x4 matrix
    fn eq(&self, other: &Matrix<4, 4, T>) -> bool {
        (0..3).flat_map(|r| (0..3).map(move |c| (r, c))).all(|(r, c)| self.rotation[r][c] == other[r][c]) // rotation
        &&
        (0..3).all(|c| self.translation[c][0] == other[c][3]) // translation
    }
}

impl<T: Float> PartialEq for Transformation<T> {
    fn eq(&self, other: &Self) -> bool {
        self.rotation == other.rotation && self.translation == other.translation
    }
}

impl<T: Float> core::fmt::Display for Transformation<T> {
    /// ```rust
    /// use seigyo::{Transformation, Matrix};
    ///
    /// let t = Transformation::from(Matrix::from([[1.], [2.], [3.]]));
    /// assert_eq!(
    ///     format!("{t}"),
    ///     "│ 1 0 0 1 │\n│ 0 1 0 2 │\n│ 0 0 1 3 │\n│ 0 0 0 1 │\n"
    /// );
    /// ```
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // caution: format macro creates heap allocated memory
        let strings: [[String; 4]; 4] = core::array::from_fn(|i| core::array::from_fn(|j|
            if i < 3 {
                if j < 3 { format!("{}", self.rotation[i][j]) }
                else { format!("{}", self.translation[i][0]) }
            }
            else if j < 3 { format!("{}", T::default()) }
            else { format!("{}", T::from_f32(1.)) }
        ));

        let col_widths: [usize; 4] = core::array::from_fn(|j| (0..4).map(|i| strings[i][j].len()).max().unwrap_or(0));

        for s in strings.iter().take(4) {
            write!(f, "│")?;
            for j in 0..4 {
                write!(f, " {:>width$}", s[j], width = col_widths[j])?;
            }
            writeln!(f, " │")?;
        }

        Ok(())
    }
}

/// Analogous to velocity
#[derive(Debug, Clone, PartialEq)]
pub struct Twist<T: Float> {
    /// represents angular velocity in rad/s
    angular: Matrix<3, 1, T>,
    /// represents linear velocity in m/s
    linear: Matrix<3, 1, T>,
}

impl<T: Float> From<Matrix<6, 1, T>> for Twist<T> {
    /// Converts a column seigyo with 6 elements to a [Twist]
    /// ```rust
    /// use seigyo::{Matrix, Twist};
    ///
    /// let mat = Matrix::from([
    ///     [12.],
    ///     [1.],
    ///     [1.],
    ///     [0.898],
    ///     [-0.2222],
    ///     [0.898],
    /// ]);
    ///
    /// let _twist: Twist<f32> = mat.into();
    /// ```
    fn from(value: Matrix<6, 1, T>) -> Self {
        Self {
            angular: Matrix::from([[value[0][0]], [value[1][0]], [value[2][0]]]),
            linear: Matrix::from([[value[3][0]], [value[4][0]], [value[5][0]]]),
        }
    }
}

impl<T: Float> From<Screw<T>> for Twist<T> {
    fn from(value: Screw<T>) -> Self {
        Self {
            angular: value.angular,
            linear: value.linear,
        }
    }
}

impl<T: Float> core::fmt::Display for Twist<T> {
    /// Display for Twist
    /// ```rust
    /// use seigyo::{Matrix, Twist};
    ///
    /// let matrix = Matrix::from([
    ///     [0.],
    ///     [0.],
    ///     [-2.],
    ///     [2.8],
    ///     [4.],
    ///     [0.],
    /// ]);
    ///
    /// assert_eq!(Twist::from(matrix).to_string(), "│\t0\t│\n│\t0\t│\n│\t-2\t│\n│\t2.8\t│\n│\t4\t│\n│\t0\t│\n");
    /// ```
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for i in 0..3 {
            writeln!(f, "│\t{}\t│", self.angular[i][0])?;
        }
        for i in 0..3 {
            writeln!(f, "│\t{}\t│", self.linear[i][0])?;
        }

        Ok(())
    }
}

impl<T: Float> Twist<T> {
    pub fn new(angular: Matrix<3, 1, T>, linear: Matrix<3, 1, T>) -> Self {
        Self {
            angular,
            linear,
        }
    }

    /// According to **Chasles-Mozzi** theorem, every rigid body displacement can be expressed as a **screw** (normalized twist) motion.
    /// 
    /// Method creates the corresponding [Transformation] matrix integrating the twist for the given amount of time.
    ///
    /// **Note:** If [Twist] is normalized, it represents a **Screw** motion. In this case the time can be considered as the angle to move (distance moved per unit time).
    pub fn exp(self, time: T) -> Transformation<T> {
        let (screw, magnitude) = self.to_screw();
        screw.exp(magnitude * time)
    }

    /// Converts the twist into a screw by normalizing it. It returns the corresponding screw and the magnitude $\theta$ corresponding to it.
    pub fn to_screw(mut self) -> (Screw<T>, T) {
        let pure_translation = self.angular.is_zero();
        let theta = match pure_translation {
            true => &self.linear,
            false => &self.angular,
        }.iter().map(|r| r[0].magnitude().powi(2)).sum::<T>().sqrt();

        // normalize angular velocity if present
        if !pure_translation {
            self.angular /= theta;
        }

        // normalize linear velocity
        self.linear /= theta;

        (Screw { angular: self.angular, linear: self.linear }, theta)
    }

    /// creates the adjoint 4x4 matrix for [Twist] by taking the ownership of the [`Twist`].
    /// ```rust
    /// use seigyo::{Matrix, Twist};
    ///
    /// let matrix = Matrix::from([
    ///     [0.],
    ///     [0.],
    ///     [-2.],
    ///     [2.8],
    ///     [4.],
    ///     [0.],
    /// ]);
    ///
    /// let twist: Twist<f64> = matrix.into();
    /// assert_eq!(twist.to_adjoint(), Matrix::from([
    ///     [0., 2., 0., 2.8],
    ///     [-2., 0., 0., 4.],
    ///     [0., 0., 0., 0.],
    ///     [0., 0., 0., 0.],
    /// ]));
    /// ```
    pub fn to_adjoint(self) -> Matrix<4, 4, T> {
        let adjoint = self.angular.skew_symmetric();
        let zero = Complex::default();

        Matrix::from([
            [adjoint[0][0], adjoint[0][1], adjoint[0][2], self.linear[0][0]],
            [adjoint[1][0], adjoint[1][1], adjoint[1][2], self.linear[1][0]],
            [adjoint[2][0], adjoint[2][1], adjoint[2][2], self.linear[2][0]],
            [zero; 4],
        ])
    }

    /// Returns true if twist is normalized (i.e. represents a screw motion).
    ///
    /// A twist is a screw if its angular part is a unit vector, or if the angular part is zero and the linear part is a unit vector (pure translation).
    ///
    /// #### Test cases
    /// 1. Rotational screw: normalized angular axis.
    /// ```rust
    /// use seigyo::{Matrix, Twist};
    ///
    /// let twist = Twist::new(
    ///     Matrix::from([[0.], [0.], [1.]]),
    ///     Matrix::from([[0.], [0.], [0.]]),
    /// );
    /// 
    /// assert!(twist.is_screw());
    /// ```
    /// 
    /// 2. Translational screw: zero angular, normalized linear axis.
    /// ```rust
    /// use seigyo::{Matrix, Twist};
    /// 
    /// let twist = Twist::new(
    ///     Matrix::from([[0.], [0.], [0.]]),
    ///     Matrix::from([[0.], [0.], [1.]]),
    /// );
    /// 
    /// assert!(twist.is_screw());
    /// ```
    ///
    /// 3. Not a screw: angular is non-zero but not normalized (norm = sqrt(2)).
    /// ```rust, should_panic
    /// use seigyo::{Matrix, Twist};
    /// 
    /// let twist = Twist::new(
    ///     Matrix::from([[1.], [1.], [0.]]),
    ///     Matrix::from([[0.], [0.], [0.]]),
    /// );
    /// assert!(twist.is_screw());
    /// ```
    /// 
    /// 4. Not a screw: angular is zero but linear is not normalized.
    /// ```rust, should_panic
    /// use seigyo::{Matrix, Twist};
    /// 
    /// let twist = Twist::new(
    ///     Matrix::from([[0.], [0.], [0.]]),
    ///     Matrix::from([[1.], [2.], [3.]]),
    /// );
    /// assert!(twist.is_screw());
    /// ```
    pub fn is_screw(&self) -> bool {
        self.angular.has_normal_columns()
            ||
        self.angular.is_zero() && self.linear.has_normal_columns()
    }
}

/// Denote screw axis.
///
/// [Screw] can be thought of another way of representing [Transformation] given the amount to move $\theta$.
///
/// For a rotation screw the angular component is a unit vector. For a pure translation screw the angular
/// component is zero and the linear component is a unit vector. The one exception is [`Screw::default()`],
/// which has both components zero and represents no motion (identity [`Transformation`]).
#[derive(Debug, Clone)]
pub struct Screw<T: Float> {
    /// the angular component of the screw motion
    angular: Matrix<3, 1, T>,
    /// - Denotes the instantaneous motion of a point on axis.
    /// - For a rotation axis passing through a point $q$ with direction $\omega$, the linear component is given by:$$-\omega \times q + h . \omega$$
    ///   here **h** is the pitch of the screw with unit m/rad.
    linear: Matrix<3, 1, T>,
}

impl<T: Float> Default for Screw<T> {
    /// The default [`Screw`] represents no motion. It corresponds to identity [`Transformation`] matrix.
    fn default() -> Self {
        Self {
            angular: Matrix::new_zero(),
            linear: Matrix::new_zero(),
        }
    }
}

impl<T: Float> core::fmt::Display for Screw<T> {
    /// Display for Twist
    /// ```rust
    /// use seigyo::{Matrix, Screw};
    ///
    /// let matrix = Matrix::from([
    ///     [0.],
    ///     [0.],
    ///     [-1.],
    ///     [1.4],
    ///     [2.],
    ///     [0.],
    /// ]);
    ///
    /// let screw = Screw::try_from(matrix).expect("invalid screw matrix");
    ///
    /// assert_eq!(screw.to_string(), "│\t0\t│\n│\t0\t│\n│\t-1\t│\n│\t1.4\t│\n│\t2\t│\n│\t0\t│\n");
    /// ```
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for i in 0..3 {
            writeln!(f, "│\t{}\t│", self.angular[i][0])?;
        }
        for i in 0..3 {
            writeln!(f, "│\t{}\t│", self.linear[i][0])?;
        }

        Ok(())
    }
}

impl<T: Float> Screw<T> {
    /// Creates a new screw from angular and linear matrices.
    ///
    /// Returns an error if the axis is not normalized:
    /// - For rotation screws (non-zero angular): the angular axis must be a unit vector.
    /// - For translation screws (zero angular): the linear axis must be a unit vector.
    ///
    /// ### Success: normalized rotation axis
    /// ```rust
    /// use seigyo::{Matrix, Screw};
    ///
    /// let screw = Screw::new(
    ///     Matrix::from([[0.], [0.], [1.]]),
    ///     Matrix::from([[0.], [0.], [0.]]),
    /// );
    ///
    /// assert!(screw.is_ok());
    /// ```
    ///
    /// ### Error: rotation axis not normalized
    /// ```rust
    /// use seigyo::{Matrix, Screw, Error, ScrewError};
    ///
    /// // angular axis [1, 1, 0] has norm sqrt(2) — not a unit vector
    /// let screw: Result<Screw<f32>, _> = Screw::new(
    ///     Matrix::from([[1.], [1.], [0.]]),
    ///     Matrix::from([[0.], [0.], [0.]]),
    /// );
    ///
    /// assert_eq!(
    ///     screw.unwrap_err(),
    ///     Error::Screw(ScrewError::RotationAxisNotNormal),
    /// );
    /// ```
    ///
    /// ### Error: translation axis not normalized
    /// ```rust
    /// use seigyo::{Matrix, Screw, Error, ScrewError};
    ///
    /// // angular is zero (pure translation) but linear [1, 1, 0] has norm sqrt(2)
    /// let screw = Screw::new(
    ///     Matrix::from([[0.], [0.], [0.]]),
    ///     Matrix::from([[1.], [1.], [0.]]),
    /// );
    ///
    /// assert_eq!(
    ///     screw.unwrap_err(),
    ///     Error::Screw(ScrewError::TranslationAxisNotNormal),
    /// );
    /// ```
    pub fn new(angular: Matrix<3, 1, T>, linear: Matrix<3, 1, T>) -> Result<Self, Error> {
        // linear component only needs to be normal when the Screw represents a translation motion (angular component is zero).
        let (axis_to_check, error) = if angular.is_zero() {
            (&linear, ScrewError::TranslationAxisNotNormal)
        }
        else {
            (&angular, ScrewError::RotationAxisNotNormal)
        };

        if axis_to_check.has_normal_columns() {
            Ok(Self { angular, linear })
        }
        else {
            Err(error.into())
        }
    }

    /// Creates a new screw representing revolute joint given the normalized axis and a point on the axis.
    ///
    /// $$v = -\omega \times q$$
    pub fn new_revolute(angular: Matrix<3, 1, T>, point: Matrix<3, 1, T>) -> Result<Self, Error> {
        if angular.has_normal_columns() {
            let linear = - angular.cross(point);
            Ok(Self { angular, linear })
        }
        else {
            Err(ScrewError::RotationAxisNotNormal.into())
        }
    }

    /// Multiplying a [`Screw`] by theta scales its angular and linear components, producing the corresponding [`Twist`].
    /// ```rust
    /// use seigyo::{Matrix, Screw, Twist};
    ///
    /// // A pure rotation screw: unit axis along Z, no linear component
    /// let screw = Screw::new(
    ///     Matrix::from([[0.], [0.], [1.]]),
    ///     Matrix::from([[0.], [0.], [0.]]),
    /// ).unwrap();
    ///
    /// // Multiplying by theta = 2.0 scales the angular component
    /// let twist = screw.to_twist(2.);
    ///
    /// let twist_result = Twist::new(
    ///     Matrix::from([[0.], [0.], [2.]]),
    ///     Matrix::from([[0.], [0.], [0.]]),
    /// );
    ///
    /// assert_eq!(twist, twist_result);
    /// ```
    ///
    /// For a pure translation screw the angular component stays zero:
    ///
    /// ```rust
    /// use seigyo::{Matrix, Screw, Twist};
    ///
    /// // A pure translation screw: no angular component, unit axis along Z
    /// let screw = Screw::new(
    ///     Matrix::from([[0.], [0.], [0.]]),
    ///     Matrix::from([[0.], [0.], [1.]]),
    /// ).unwrap();
    ///
    /// // Multiplying by theta = 3.0 scales only the linear component
    /// let twist = screw.to_twist(3.);
    /// 
    /// let twist_result = Twist::new(
    ///     Matrix::from([[0.], [0.], [0.]]),
    ///     Matrix::from([[0.], [0.], [3.]]),
    /// );
    ///
    /// assert_eq!(twist, twist_result);
    /// ```
    #[inline]
    pub fn to_twist(mut self, theta: T) -> Twist<T> {
        if !self.angular.is_zero() {
            self.angular *= theta;
        }

        self.linear *= theta;
        self.into()
    }

    /// According to **Chasles-Mozzi** theorem, every rigid body displacement can be expressed as a **screw** (normalized twist) motion.
    /// 
    /// Method creates the corresponding [Transformation] matrix integrating the screw for the given amount of theta.
    pub fn exp(self, theta: T) -> Transformation<T> {
        // pure translation
        if self.angular.is_zero() {
            return Transformation { rotation: Matrix::new_identity(), translation: self.linear * theta };
        }

        // the rotation matrix is obtained from rodrigues formula while the complete matrix translation matrix is created using Chasles-Mozzi theorem.
        // 1. calculate skew symmetric matrix
        let skew = self.angular.skew_symmetric();
        let skew_sq = &skew * &skew;
        let one_minus_cos = Complex::from(T::from_f32(1.) - theta.cos());
        let sin = theta.sin();

        // 2. Rodrigues' formula
        let rotation = Matrix::new_identity() + Complex::from(sin) * skew.clone() + one_minus_cos * skew_sq.clone();

        // 3. Translation matrix
        let translation = (Matrix::new_identity() * theta + one_minus_cos * skew + Complex::from(theta - sin) * skew_sq) * self.linear;

        Transformation { rotation, translation }
    }

    /// Transforms a screw matrix given a transformation matrix
    ///
    /// $$S_a = [Ad_{T_{ab}}]S_b$$
    pub fn transform(self, transformation: Transformation<T>) -> Self {
        let adjoint = transformation.to_adjoint();
        let matrix: Matrix<6, 1, T> = self.into();

        let value = adjoint * matrix;

        let angular = Matrix::from([
            value[0],
            value[1],
            value[2],
        ]);

        let linear = Matrix::from([
            value[3],
            value[4],
            value[5],
        ]);

        Self{ angular, linear }
    }

    /// creates the adjoint 4x4 matrix for [`Screw`] by taking the ownership of the [`Screw`].
    /// ```rust
    /// use seigyo::{Matrix, Screw};
    ///
    /// let matrix = Matrix::from([
    ///     [0.],
    ///     [0.],
    ///     [-1.],
    ///     [1.4],
    ///     [2.],
    ///     [0.],
    /// ]);
    ///
    /// let screw = Screw::try_from(matrix).expect("invalid screw matrix");
    /// 
    /// assert_eq!(screw.to_adjoint(), Matrix::from([
    ///     [0., 1., 0., 1.4],
    ///     [-1., 0., 0., 2.],
    ///     [0., 0., 0., 0.],
    ///     [0., 0., 0., 0.],
    /// ]));
    /// ```
    pub fn to_adjoint(self) -> Matrix<4, 4, T> {
        let adjoint = self.angular.skew_symmetric();
        let zero = Complex::default();

        Matrix::from([
            [adjoint[0][0], adjoint[0][1], adjoint[0][2], self.linear[0][0]],
            [adjoint[1][0], adjoint[1][1], adjoint[1][2], self.linear[1][0]],
            [adjoint[2][0], adjoint[2][1], adjoint[2][2], self.linear[2][0]],
            [zero; 4],
        ])
    }
}

impl<T: Float> TryFrom<Matrix<6, 1, T>> for Screw<T> {
    type Error = Error;

    fn try_from(value: Matrix<6, 1, T>) -> Result<Self, Error> {
        let angular = Matrix::from([
            value[0],
            value[1],
            value[2],
        ]);
        let linear = Matrix::from([
            value[3],
            value[4],
            value[5],
        ]);

        Self::new(angular, linear)
    }
}

impl<T: Float> From<Screw<T>> for Matrix<6, 1, T> {
    fn from(value: Screw<T>) -> Self {
        let a = value.angular;
        let l = value.linear;

        Self::from([
            [a[0][0]],
            [a[1][0]],
            [a[2][0]],
            [l[0][0]],
            [l[1][0]],
            [l[2][0]],
        ])
    }
}

/// Multiplying a [`Screw`] by a theta gives a [`Twist`].
impl<T: Float> core::ops::Mul<T> for Screw<T> {
    type Output = Twist<T>;

    fn mul(self, rhs: T) -> Self::Output {
        self.to_twist(rhs)
    }
}

/// # Moment of Inertia
/// Describes the $3 \times 3$ moment of inertia [`Matrix`].
///
/// Note that all fields are stored in the *center-of-mass frame* where the object axes are aligned with the principal axes of inertia.
///
/// ### Todo
/// - Generalize this to store transformed matrix. (all 9 elements).
/// - Should be able to add and subtract objects.
/// - This could mean finding new COM and new principal axes.
/// - This could mean storing two transformations:
///     1. $T\_{com}$: COM position in world frame {s}.
///     2. $T_\text{inertia}$: Inertia axes in object frame {b}.
/// - Once objects are combined, their inertia frame would be common and their world frame would merge.
///     - Should the world frame align with the principal axes? This would mean finding eigen vectors of the inertia matrix.
#[derive(Debug, Default, Clone)]
pub struct Inertia<T: Float> {
    /// mass of the object
    ///
    /// this is used in the method [`Inertia::transform`].
    m: T,
    ixx: T, iyy: T, izz: T,
    ixy: T, iyz: T, ixz: T,
    /// transformation of the current inertia frame in the world frame
    t: Transformation<T>,
}

impl<T: Float> Inertia<T> {
    /// - Cretes a new [`Inertia`] matrix from the given mass and the principal inertial components $i\_{xx}$, $i\_{yy}$ and $i_\{zz}$.
    pub fn new(m: T, inertia_matrix: Matrix<3, 3, T>, maybe_t: Option<Transformation<T>>) -> Result<Self, Error> {
        if inertia_matrix.is_symmetric() {
            let [ixx, iyy, izz] = core::array::from_fn(|i| inertia_matrix[i][i].real());
            let ixy = inertia_matrix[0][1].real();
            let iyz = inertia_matrix[1][2].real();
            let ixz = inertia_matrix[0][2].real();

            Ok(Self {
                m,
                ixx, iyy, izz,
                ixy, iyz, ixz,
                t: maybe_t.unwrap_or_default(),
            })
        }
        else {
            Err(Error::AsymmetricMatrix)
        }
    }

    /// ## Moment of inertia of Cuboid
    /// Assumptions:
    /// - The mass density is uniform.
    /// - The x, y, z align with the **principal axes of the object**. The axes passes through the center-of-mass of the shape.
    ///
    /// The formula is given by:
    /// $$\mathcal{I}_w = \frac{m}{12}\left( l^2 + h^2 \right)$$
    /// 
    /// ```rust
    /// use seigyo::{Matrix, Inertia};
    /// 
    /// let moi = Inertia::new_cuboid(24., 0.01, 0.054, 0.033);
    /// 
    /// assert_eq!(Matrix::from(moi), Matrix::from([
    ///     [0.00801, 0., 0.],
    ///     [0., 0.0023780000000000003, 0.],
    ///     [0., 0., 0.006031999999999999],
    /// ]));
    /// ```
    pub fn new_cuboid(m: T, x: T, y: T, z: T) -> Self {
        let [x_sq, y_sq, z_sq] = [x, y, z].map(|a| a.powi(2));

        let moi = |a_sq: T, b_sq: T| m * (a_sq + b_sq) / T::from_f32(12.);

        let ixx = moi(y_sq, z_sq);
        let iyy = moi(x_sq, z_sq);
        let izz = moi(x_sq, y_sq);

        Self { m, ixx, iyy, izz, ..Default::default() }
    }

    /// ## Moment of inertia of Cylinder
    /// Assumptions:
    /// - The mass density is uniform.
    /// - The x, y, z align with the **principal axes of the object**. The axes passes through the centroid of the shape.
    /// - The cylinder axis is the *z axis*.
    /// ```rust
    /// use seigyo::{Matrix, Inertia};
    /// 
    /// let moi = Inertia::new_cylinder(1.2, 0.058, 0.012);
    /// 
    /// assert_eq!(Matrix::from(moi), Matrix::from([
    ///     [0.0010236, 0., 0.],
    ///     [0., 0.0010236, 0.],
    ///     [0., 0., 0.0020184],
    /// ]));
    /// ```
    pub fn new_cylinder(m: T, r: T, h: T) -> Self {
        let r_sq = r.powi(2);

        let moi_curved = m * (T::from_f32(3.) * r_sq + h.powi(2)) / T::from_f32(12.);
        let izz = m * r_sq / T::from_f32(2.);

        Self { m, ixx: moi_curved, iyy: moi_curved, izz, ..Default::default() }
    }
    
    /// ## Moment of inertia of Ellepsoid
    /// Assumptions:
    /// - The mass density is uniform
    /// - The x, y, z axes align with the **principal axes of the object**. The axes passes though the centroid of the shape.
    /// ```rust
    /// use seigyo::{Matrix, Inertia};
    ///
    /// let moi = Inertia::new_ellepsoid(21.53, 0.080, 0.120, 0.081);
    ///
    /// assert_eq!(Matrix::from(moi), Matrix::from([
    ///     [0.090258066, 0., 0.],
    ///     [0., 0.055810066000000005, 0.],
    ///     [0., 0., 0.0895648],
    /// ]));
    /// ```
    pub fn new_ellepsoid(m: T, x: T, y: T, z: T) -> Self {
        let [x_sq, y_sq, z_sq] = [x, y, z].map(|a| a.powi(2));

        let moi = |a_sq: T, b_sq: T| m * (a_sq + b_sq) / T::from_f32(5.);

        let ixx = moi(y_sq, z_sq);
        let iyy = moi(x_sq, z_sq);
        let izz = moi(x_sq, y_sq);

        Self { m, ixx, iyy, izz, ..Default::default() }
    }

    /// ## Transformation of Inertia Martix
    /// ### Rotation
    /// We can describe the inertia matrix $\mathcal{I}_b$ in a rotated frame {c}, where the rotation matrix is given by $R\_{bc}$ as follows:
    /// $$\mathcal{I}_c = R\_{bc}^T \mathcal{I}_b R\_{bc}$$
    ///
    /// **Note:** The proof is given by considering the fact that *kinetic energy* remains same in any frame.
    ///
    /// ### Translation
    /// - The translation of an Inertia matrix is given by **Steiner's Theorem**.
    /// - The inertia matrix at a point $q = (q_x, q_y, q_z)$ in {b} (body frame) calculated at the center of mass is given by:
    ///   $$\mathcal{I}_q = \mathcal{b} + \mathbf{m}\left( q^TqI - qq^T \right)$$
    /// - The term $\mathcal{m} \left( q^TqI - qq^T \right)$ is always positive semi definite. This means we are adding a 
    ///
    /// ```rust
    /// todo!()
    /// ```
    pub fn transform_to(&mut self, t_new: Transformation<T>) {
        let Inertia { m, ixx, iyy, izz, ixy, iyz, ixz, t: t_current } = self;
        let i = Matrix::from([
            [ixx.to_owned(), ixy.to_owned(), ixz.to_owned()],
            [ixy.to_owned(), iyy.to_owned(), iyz.to_owned()],
            [ixz.to_owned(), iyz.to_owned(), izz.to_owned()],
        ]);

        // find the new transformation
        // current is T_{cur, world}
        // t is T_{new, world}
        // to find T_{new, cur}
        // T_{new, cur} = T_{new, world} * T_{world, cur} = T_{new, world} * T_{cur, world}.inverse
        let Transformation { rotation, translation } = t_new.clone() * t_current.to_owned().inverse();

        let i_ref = rotation.clone().transpose() * i * rotation;

        // Steiner's theorem
        let tra_trat = &translation * translation.clone().transpose();
        let tra_normsq = translation.iter().map(|r| r[0].real().powi(2)).sum::<T>();
        [*ixx, *iyy, *izz] = core::array::from_fn(|i| i_ref[i][i].real() + (tra_normsq - tra_trat[i][i].real()) * *m);
        let off_diagonal = |x: usize, y: usize| i_ref[x][y].real() - tra_trat[x][y].real() * *m;
        *ixy = off_diagonal(0, 1);
        *ixz = off_diagonal(0, 2);
        *iyz = off_diagonal(1, 2);

        *t_current = t_new;
    }
}

impl<T: Float> From<Inertia<T>> for Matrix<3, 3, T> {
    fn from(Inertia { ixx, iyy, izz, ixy, iyz, ixz, .. }: Inertia<T>) -> Self {
        Self::from([
            [ixx, ixy, ixz],
            [ixy, iyy, iyz],
            [ixz, iyz, izz],
        ])
    }
}

impl<T: Float> core::ops::Add for Inertia<T> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<T: Float> core::ops::AddAssign for Inertia<T> {
    /// ```rust
    /// todo!()
    /// ```
    #[inline]
    fn add_assign(&mut self, mut rhs: Self) {
        rhs.transform_to(self.t.clone());

        self.m += rhs.m;
        self.ixx += rhs.ixx;
        self.iyy += rhs.iyy;
        self.izz += rhs.izz;
        self.ixy += rhs.ixy;
        self.iyz += rhs.iyz;
        self.ixz += rhs.ixz;
    }
}

impl<T: Float> core::ops::Sub for Inertia<T> {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl<T: Float> core::ops::SubAssign for Inertia<T> {
    /// ```rust
    /// todo!()
    /// ```
    #[inline]
    fn sub_assign(&mut self, mut rhs: Self) {
        rhs.transform_to(self.t.clone());

        self.m += rhs.m;
        self.ixx -= rhs.ixx;
        self.iyy -= rhs.iyy;
        self.izz -= rhs.izz;
        self.ixy -= rhs.ixy;
        self.iyz -= rhs.iyz;
        self.ixz -= rhs.ixz;
    }
}

/// Trait implements forward and backward kinematics for manipulator arms.
pub trait KinematicsChain<const N: usize, T: Float = f64> {
    /// ## Forward kinematics in body frame.
    /// - **m** represents the end-effector configuration, when root is at home position.
    /// 
    /// **Reference:** Example 4.7 from Modern Robotics
    /// ```rust
    /// use seigyo::{Matrix, Transformation, Screw, KinematicsChain};
    /// use core::f32::consts::PI;
    ///
    /// struct Manipulator;
    ///
    /// impl KinematicsChain<7, f32> for Manipulator{}
    ///
    /// let m = Transformation::try_from(Matrix::from([
    ///     [1., 0., 0., 0.],
    ///     [0., 1., 0., 0.],
    ///     [0., 0., 1., 0.060 + 0.3 + 0.55],
    ///     [0., 0., 0., 1.],
    /// ])).unwrap();
    ///
    /// let b_list = [
    ///     Screw::new(Matrix::from([[0.], [0.], [1.]]), Matrix::new_zero()).unwrap(),
    ///     Screw::new(Matrix::from([[0.], [1.], [0.]]), Matrix::from([[0.060 + 0.3 + 0.55], [0.], [0.]])).unwrap(),
    ///     Screw::new(Matrix::from([[0.], [0.], [1.]]), Matrix::new_zero()).unwrap(),
    ///     Screw::new(Matrix::from([[0.], [1.], [0.]]), Matrix::from([[0.060 + 0.3], [0.], [0.045]])).unwrap(),
    ///     Screw::new(Matrix::from([[0.], [0.], [1.]]), Matrix::from([[0.], [0.], [0.]])).unwrap(),
    ///     Screw::new(Matrix::from([[0.], [1.], [0.]]), Matrix::from([[0.060], [0.], [0.]])).unwrap(),
    ///     Screw::new(Matrix::from([[0.], [0.], [1.]]), Matrix::from([[0.], [0.], [0.]])).unwrap(),
    /// ];
    ///
    /// let theta_list = [ 0., PI / 4., 0., -PI / 4., 0., -PI / 2., 0. ];
    ///
    /// let manipulator = Manipulator;
    /// let fk = manipulator.fk_body(m, b_list, theta_list);
    ///
    /// assert_eq!(fk, Matrix::from([
    ///     [0., 0., -0.99999994, 0.31572855],
    ///     [0., 1., 0., 0.],
    ///     [0.99999994, 0., 0., 0.65708894],
    ///     [0., 0., 0., 1.]
    /// ]));
    /// ```
    fn fk_body(&self, m: Transformation<T>, b_list: [Screw<T>; N], theta_list: [T; N]) -> Transformation<T> {
        b_list.into_iter().zip(theta_list).fold(m, |acc, (b_frame, theta)| acc * b_frame.exp(theta))
    }
    
    /// Forward kinematics in space frame
    /// ```rust
    /// use seigyo::{Matrix, Transformation, Screw, KinematicsChain};
    /// use core::f32::consts::PI;
    ///
    /// struct Manipulator;
    ///
    /// impl KinematicsChain<6, f32> for Manipulator {}
    /// 
    /// let m = Transformation::try_from(Matrix::from([
    ///     [-1., 0., 0., 0.425 + 0.392],
    ///     [0., 0., 1., 0.109 + 0.082],
    ///     [0., 1., 0., 0.089 - 0.095],
    ///     [0., 0., 0., 1.],
    /// ])).unwrap();
    ///
    /// let s_list = [
    ///     Screw::new(Matrix::from([[0.], [0.], [1.]]), Matrix::new_zero()).unwrap(),
    ///     Screw::new(Matrix::from([[0.], [1.], [0.]]), Matrix::from([[-0.089], [0.], [0.]])).unwrap(),
    ///     Screw::new(Matrix::from([[0.], [1.], [0.]]), Matrix::from([[-0.089], [0.], [0.425]])).unwrap(),
    ///     Screw::new(Matrix::from([[0.], [1.], [0.]]), Matrix::from([[-0.089], [0.], [0.425 + 0.392]])).unwrap(),
    ///     Screw::new(Matrix::from([[0.], [0.], [-1.]]), Matrix::from([[-0.109], [0.425 + 0.392], [0.]])).unwrap(),
    ///     Screw::new(Matrix::from([[0.], [1.], [0.]]), Matrix::from([[0.095 - 0.089], [0.], [0.425 + 0.392]])).unwrap(),
    /// ];
    ///
    /// let theta_list = [ 0., -PI / 2., 0., 0., PI / 2., 0. ];
    ///
    /// let manipulator = Manipulator;
    /// let fk = manipulator.fk_space(m, s_list, theta_list);
    /// println!("fk:\n{fk}");
    ///
    /// assert_eq!(fk, Matrix::from([
    ///     [0., -1., 0., 0.095],
    ///     [1., 0., 0., 0.10899997],
    ///     [0., 0., 1., 0.98800004],
    ///     [0., 0., 0., 1.],
    /// ]));
    /// ```
    fn fk_space(&self, m: Transformation<T>, s_list: [Screw<T>; N], theta_list: [T; N]) -> Transformation<T> {
        s_list.into_iter().zip(theta_list).rev().fold(m, |acc, (s_frame, theta)| s_frame.exp(theta) * acc)
    }

    fn j_body(b_list: [Screw<T>; N], theta_list: [T; N]) -> Matrix<6, N> {
        todo!("implement body jacobian")
    }

    /// Returns the space jacobian given the following arguments:
    /// - The [`Screw`] representation of the axes.
    /// - The current state of the robot.
    ///
    /// ```rust
    /// todo!()
    /// ```
    fn j_space(s_list: [Screw<T>; N], theta_list: [T; N]) -> Matrix<6, N, T> {
        let mut j = Matrix::new_zero();
        let mut t = Transformation::default();

        for (c, (screw, theta)) in s_list.into_iter().zip(theta_list).enumerate() {
            let j_i = t.clone().to_adjoint() * Matrix::from(screw.clone());
            j.iter_mut().zip(j_i.iter()).for_each(|(row, ji_row)| row[c] = ji_row[0]);
            t *= screw.exp(theta);
        }

        j
    }
}

#[derive(Debug, PartialEq)]
pub enum Error {
    /// Error for matrix operations
    SingularMatrix,
    /// [`Screw`] errors
    Screw(ScrewError),
    /// [`Transformation`] errors
    Transformation(TransformationError),
    /// [`Inertia`] error
    AsymmetricMatrix,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::SingularMatrix => write!(f, "matrix is singular"),
            Self::Transformation(e) => write!(f, "error while making transformation matrix. {e}"),
            Self::Screw(e) => write!(f, "error while making screw matrix. {e}"),
            Self::AsymmetricMatrix => write!(f, "matrix is not a symmetric matrix"),
        }
    }
}

impl From<TransformationError> for Error {
    fn from(value: TransformationError) -> Self {
        Self::Transformation(value)
    }
}

impl From<ScrewError> for Error {
    fn from(value: ScrewError) -> Self {
        Self::Screw(value)
    }
}

#[derive(Debug, PartialEq)]
pub enum TransformationError {
    /// The provided rotation matrix is singular
    RotationSingular,
    /// The provided rotation matrix doesn't have orthogonal columns
    RotationNonOrthogonal,
    /// The rotation matrix has invalid orientation
    RotationInvalidOrientation,
}

impl core::fmt::Display for TransformationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RotationSingular => write!(f, "singular matrix received while creating rotation matrix"),
            Self::RotationNonOrthogonal => write!(f, "columns of rotation matrix are not orthogonal"),
            Self::RotationInvalidOrientation => write!(f, "rotation matrix not right handed. this represents reflection matrix."),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum ScrewError {
    RotationAxisNotNormal,
    TranslationAxisNotNormal,
}

impl core::fmt::Display for ScrewError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::RotationAxisNotNormal => write!(f, "screw axis should be normalized"),
            Self::TranslationAxisNotNormal => write!(f, "translation vector should be normal if there is no rotation (theta defines the magnitude of translation).")
        }
    }
}

impl core::error::Error for ScrewError {}
