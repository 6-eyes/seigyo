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
{
    /// abstracting square root from floating types
    fn sqrt(self) -> Self;

    /// abstracting powi from floating types
    fn powi(self, n: i32) -> Self;

    /// method to produce zero vector for the floating type
    fn zero() -> Self;

    /// method to produce unity vector for the floating type
    fn one() -> Self;

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

    /// the minimum tolerated value.
    /// any value less than this will be considered zero
    fn tolerance() -> Self;

    /// abstracting sin from floating types
    fn sin(self) -> Self;

    /// abstracting cos from floating types
    fn cos(self) -> Self;
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
    fn zero() -> Self {
        0.
    }

    #[inline(always)]
    fn one() -> Self {
        1.
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
    fn zero() -> Self {
        0.
    }

    #[inline(always)]
    fn one() -> Self {
        1.
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
}

#[derive(Debug, Clone, Copy)]
pub struct Complex<T: Float = f32>(T, T);

impl<T: Float> Default for Complex<T> {
    fn default() -> Self {
        Self(T::zero(), T::zero())
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
    /// use std::f32::consts::PI;
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
            self.norm_squared().sqrt()
        }
    }

    /// returns the square of the norm
    /// #### Motivation
    /// - this can be used in multiple place where the equation demands square of a norm.
    /// - this can be used to avoid the step of calculating square root and then calculating the square again.
    #[inline(always)]
    pub fn norm_squared(&self) -> T {
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
            (false, false) => self / self.norm_squared().sqrt(),
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
        let mag = rhs.norm_squared();
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
        Self(var, T::zero())
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
impl<T: Float> std::fmt::Display for Complex<T> {
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
   fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        if self.0 == T::zero() {
            write!(f, "{}", if self.1 == T::zero() { "0".to_string() } else { format!("{}j", self.1) })?;
        }
        else {
            write!(f, "{}", self.0)?;
            if self.1 != T::zero() {
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

#[derive(Debug)]
pub enum MatrixError {
    Singular,
}

impl std::fmt::Display for MatrixError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Singular => write!(f, "Matrix is singular"),
        }
    }
}

impl std::error::Error for MatrixError {}

/// Defines a matrix
#[derive(Debug, PartialEq, Clone)]
pub struct Matrix<const R: usize, const C: usize, T: Float = f32>([[Complex<T>; C]; R]);

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
    /// assert_eq!("│\t10\t0\t20\t│\n│\t0\t30\t0\t│\n│\t200\t0\t100\t│\n", matrix.to_string());
    /// ```
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for i in 0..R {
            write!(f, "│\t")?;
            for j in 0..C {
                write!(f, "{}\t", self.0[i][j])?;
            }
            writeln!(f, "│")?;
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
        let matrix = std::array::from_fn(|i| // self row
            std::array::from_fn(|j| // rhs column
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
        let matrix = std::array::from_fn(|i| // row in self
            std::array::from_fn(|j| // col in rhs
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
        Self([[T::zero().into(); C]; R])
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
    pub fn transpose(&self) -> Matrix<C, R, T> {
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
    /// $O(R \times C \times \mbox{rank})$
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
    pub fn round(mut self) -> Self {
        self.0.iter_mut().for_each(|r| r.iter_mut().for_each(|c| *c = c.round()));
        self
    }

    /// ## QR Decomposition
    /// Decomposes the given matrix into two matrices using QR decomposition.
    /// The first matrix Q is a **Unitary matrix**. Meaning, it's conjugate transpose is equal to it's inverse.$$Q^{\dagger} = Q^{-1}$$
    /// The QR decomposition performed here uses Householder reflections because it is more numerically stable than gaussian eliminations.
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
    /// let (q, r) = a.qr().unwrap();
    ///
    /// assert_eq!(q, Matrix::from([[-0.857142857142857, 0.394285714285714, 0.3314285714285714], [-0.42857142857142855, -0.9028571428571422, -0.03428571428571427], [0.2857142857142857, -0.17142857142857137, 0.942857142857143]]));
    /// assert_eq!(r, Matrix::from([[-13.999999999999998, -21.000000000000004, 14.000000000000002], [-0.0000000000000007783517420333592, -174.9999999999999, 69.99999999999996], [-0.0000000000000005335902659890752, 0.000000000000007105427357601002, -35.]]));
    /// ```
    ///
    /// ```rust, should_panic
    /// use seigyo::Matrix;
    ///
    /// // Dependent columns
    /// let a = Matrix::from([[3., 3.], [1., 1.]]);
    /// a.qr().unwrap();
    /// ```
    pub fn qr(&self) -> Result<(Matrix<R, R, T>, Matrix<R, C, T>), MatrixError> {
        // todo: for m < n
        let mut r = self.to_owned();
        let mut q = Matrix::new_identity();

        let two = T::one() + T::one();
        for i in 0..C {
            // norm of i-th column
            let norm_x_squared = (i..R).map(|k| r[k][i].norm_squared()).sum::<T>();
            let norm = norm_x_squared.sqrt();
            if norm < T::tolerance() {
                return Err(MatrixError::Singular);
            }

            let alpha = match r[i][i].is_zero() {
                true => norm.into(),
                false => -(r[i][i].normalize() * norm),
            };

            // u normalized is:
            // sqrt((x_1 - alpha)^2 + x_2^2 + x_3^2 + ...)
            // sqrt(alpha^2 - 2 * x_1 * alpha + x_1^2 + x_2^2 + x_3^2 + ...)
            // sqrt(alpha^2 - 2 * x_1 * alpha + norm_x_squared)
            let u_norm = (norm_x_squared + alpha.norm_squared() - two * r[i][i].dot(alpha)).sqrt();

            if u_norm.abs() < T::tolerance() {
                // no normalization is required
                // householder matrix is identity
                // no change in r and q
                continue;
            }

            let mut u_cache: [Option<Complex<T>>; R] = core::array::from_fn(|_| None);
            let mut u = |k: usize| match u_cache[k] { // k ranges from i..R
                Some(res) => res,
                None => {
                    let mut u_k = r[k][i];
                    if k == i {
                        u_k -= alpha;
                    }

                    let res = u_k / u_norm;
                    u_cache[k] = Some(res);
                    res
                },
            };

            // create q
            let mut q_loc = Matrix::new_identity();
            (i..R).flat_map(|row| (i..C).map(move |col| (row, col))).for_each(|(row, col)| q_loc[row][col] -= u(row) * u(col) * two);

            r = &q_loc * r;
            q *= q_loc.conjugate_transpose();
        }
        
        Ok((q, r))
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
            let sum = (0..R).map(|r| self.0[r][c].norm_squared()).sum::<T>().sqrt();
            (0..R).for_each(|r| self[r][c] /= sum);
        }
    }

    /// ## Normal columns check
    /// Returns true if the columns of the matrix are normalized.
    /// 
    /// Exact `1` is unrealistic. Tolerance is used to determine how close the sum is to `1`.
    pub fn has_normal_columns(&self) -> bool {
        (0..C).map(|c| // cols
            (0..R).map(|r| self.0[r][c].norm_squared()).sum::<T>().sqrt() // sum all elements in the row
        ).all(|sum| (sum - T::one()).abs() <= T::tolerance())
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
        Self(std::array::from_fn(|i| 
                std::array::from_fn(|j| match j == i {
                    true => T::one(),
                    false => T::zero(),
                }.into())
        ))
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
        let [mut det, mut total] = [Complex::from(T::one()); 2];

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
            let temp: [Complex<T>; C] = std::array::from_fn(|j| matrix[diag_row][j]);
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
    /// const PRECISION: f32 = 1e2;
    ///
    /// let a = Matrix::from([[2., -1., 0.], [-1., 2., -1.], [0., -1., 2.]]);
    /// let inverse = {
    ///     let matrix = a.inverse().unwrap();
    ///     (matrix * PRECISION).round() / PRECISION
    /// };
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
    pub fn inverse(&self) -> Result<Self, MatrixError> {
        let mut matrix = self.0;
        let mut inverse = Self::new_identity();

        for i in 0..C {
            let max_row = match (i..C).max_by(|&a, &b| {
                let magnitude = |z: Complex<T>| z.real().abs() + z.imaginary().abs();
                magnitude(matrix[a][i]).partial_cmp(&magnitude(matrix[b][i])).unwrap_or(core::cmp::Ordering::Equal)
            }) {
                Some(max_row) if !matrix[max_row][i].is_zero() => max_row,
                _ => return Err(MatrixError::Singular),
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
}

impl<T: Float> Matrix<3, 1, T> {
    /// Method to make skew symmetric of a 3x1 matrix
    fn skew_symmetric(&self) -> Matrix<3, 3, T> {
        let zero = T::zero().into();
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

pub mod statics {
    use core::fmt::Display;
    use crate::{Matrix, Complex, Float};

    type Result<T> = core::result::Result<T, Error>;

    /// # Transformation
    /// Transformation composes of the rotation and translation components.
    /// 
    /// **Uses:**
    /// - To represent configuration of a rigid body with respect to a fixed frame represented by [`Transformation::default()`].
    /// - To change the reference frame in which a vector or frame is represented.
    /// - To displace a vector or frame.
    #[derive(Debug)]
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
        /// use seigyo::{statics::Transformation, Matrix};
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
        /// use seigyo::{statics::Transformation, Matrix};
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
        /// use seigyo::{statics::Transformation, Matrix};
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
        /// ### Transformation inverse
        /// Method to find inverse of the transformation matrix
        /// $$T^{-1} = \begin{bmatrix} R & p \\ 0 & 1 \end{bmatrix}^{-1} = \begin{bmatrix} R^T & -R^Tp \\ 0 & 1\end{bmatrix}$$
        /// Also,$$R^TR = I$$
        pub fn inverse(&self) -> Self {
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
        fn validate_rotation(rotation: &Matrix<3, 3, T>) -> Result<()> {
            // matrix should not be singular
            if rotation.is_singular() {
                return Err(Error::RotationSingular);
            }

            // columns should be orthogonal
            if [(0, 1), (0, 2), (1, 2)].iter().any(|&(c1, c2)| !(0..3).map(|i| rotation[i][c1] * rotation[i][c2]).sum::<Complex<T>>().is_zero()) {
                return Err(Error::RotationNonOrthogonal);
            }

            // check orientation: determinant should be real and equal to 1
            let determinant = rotation.determinant();
            if determinant.is_imaginary() || (determinant.real() - T::one()).abs() > T::tolerance() {
                return Err(Error::RotationInvalidOrientation);
            }

            Ok(())
        }


        /// ### Transformation adjoint
        /// Returns the 6x6 adjoint matrix for the transformation matrix
        pub fn adjoint(&self) -> Matrix<6, 6, T> {
            let one_zero = self.translation.skew_symmetric() * &self.rotation;
            let zero = T::zero().into();

            Matrix::from([
                [self.rotation[0][0], self.rotation[0][1], self.rotation[0][2], zero, zero, zero],
                [self.rotation[1][0], self.rotation[1][1], self.rotation[1][2], zero, zero, zero],
                [self.rotation[2][0], self.rotation[2][1], self.rotation[2][2], zero, zero, zero],
                [one_zero[0][0], one_zero[0][1], one_zero[0][2], self.rotation[0][0], self.rotation[0][1], self.rotation[0][2]],
                [one_zero[1][0], one_zero[1][1], one_zero[1][2], self.rotation[1][0], self.rotation[1][1], self.rotation[1][2]],
                [one_zero[2][0], one_zero[2][1], one_zero[2][2], self.rotation[2][0], self.rotation[2][1], self.rotation[2][2]],
            ])
        }

        // pub fn from_axis_and_angle(screw: Screw<T>, theta: T) -> Self {
        //     (&(screw, theta)).into()
        // }
    }

    // impl<T: Float> From<&(Screw<T>, T)> for Transformation<T> {
    //     /// ### Example 2
    //     /// Rotation matrix is non zero. Theta here denotes the angle to be rotated about the axis.
    //     /// 
    //     /// **Note:** There is no need to check the validity of rotation matrix since angular component is of [Screw] is always normalized.
    //     /// ```rust
    //     /// use seigyo::{statics::{Transformation, Screw}, Matrix};
    //     ///
    //     /// let screw = Screw::new(
    //     ///     Matrix::from([[0.], [1.], [0.]]),
    //     ///     Matrix::from([[-0.089], [0.], [0.]]),
    //     /// ).unwrap();
    //     ///
    //     /// let theta = -core::f32::consts::PI / 2.;
    //     /// let transformation = Transformation::from(&(screw, theta));
    //     ///
    //     /// assert_eq!(transformation, Matrix::from([
    //     ///     [0., 0., -1., 0.089],
    //     ///     [0., 1., -0., 0.],
    //     ///     [1., 0., -0., 0.089],
    //     ///     [0., 0., -0., 0.],
    //     /// ]));
    //     /// ```
    //     fn from((Screw { angular, linear }, theta): &(Screw<T>, T)) -> Self {
    //         // no movement
    //         let mut rotation = Matrix::new_identity();
    //         // infinite pitch
    //         let translation = if angular.is_zero() {
    //             linear.to_owned() * theta
    //         }
    //         // finite pitch
    //         else {
    //             // the rotation matrix is obtained from rodrigurs formula while the complete matrix translation matrix is created using Chasles-Mozzi theorem.
    //             // 1. calculate skew symmetric matrix
    //             let skew_symmetric = angular.skew_symmetric();
    //             let skew_symmetric_squared = &skew_symmetric * &skew_symmetric;

    //             let one_minus_cos = Complex::from(T::one() - theta.cos());
    //             let sin = theta.sin();

    //             // 2. Rodrigues' formula
    //             rotation += Complex::from(sin) * skew_symmetric.to_owned() + one_minus_cos * skew_symmetric_squared.to_owned();

    //             // 3. Translation matrix
    //             (Matrix::new_identity() * theta + one_minus_cos * skew_symmetric + Complex::from(*theta - sin) * skew_symmetric_squared) * linear
    //         };

    //         Self {
    //             rotation,
    //             translation,
    //         }
    //     }
    // }

    impl<T: Float> TryFrom<Matrix<3, 3, T>> for Transformation<T> {
        type Error = Error;

        /// New transformation matrix from a rotation matrix
        /// ```rust
        /// todo!("add test cases")
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
        /// todo!("add test cases")
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

    impl<T: Float> TryFrom<(Matrix<3, 3, T>, Matrix<3, 1, T>)> for Transformation<T> {
        type Error = Error;

        /// Transformation matrix from a tuple containing a 3x3 rotation matrix and a 3x1 translation matrix.
        /// ```rust
        /// todo!("add test cases")
        /// ```
        fn try_from((rotation, translation): (Matrix<3, 3, T>, Matrix<3, 1, T>)) -> core::result::Result<Self, Self::Error> {
            Self::validate_rotation(&rotation)?;

            Ok(Self {
                rotation,
                translation,
            })
        }
    }

    impl<T: Float> From<Matrix<3, 1, T>> for Transformation<T> {
        /// Transformation matrix from a translation matrix
        /// ```rust
        /// use seigyo::{statics::Transformation, Matrix};
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

    impl<T: Float> Display for Transformation<T> {
        /// ```rust
        /// use seigyo::{statics::Transformation, Matrix};
        ///
        /// let t = Transformation::from(Matrix::from([[1.], [2.], [3.]]));
        /// assert_eq!(
        ///     format!("{t}"),
        ///     "│\t1\t0\t0\t1\t│\n│\t0\t1\t0\t2\t│\n│\t0\t0\t1\t3\t│\n│\t0\t0\t0\t1\t│\n"
        /// );
        /// ```
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            for i in 0..3 {
                write!(f, "│\t")?;
                for j in 0..3 {
                    write!(f, "{}\t", self.rotation[i][j])?;
                }
                writeln!(f, "{}\t│", self.translation[i][0])?;
            }

            writeln!(f, "│\t{0}\t{0}\t{0}\t{1}\t│", T::zero(), T::one())
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
        /// use seigyo::{Matrix, statics::Twist};
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

    impl<T: Float> Display for Twist<T> {
        /// Display for Twist
        /// ```rust
        /// use seigyo::{Matrix, statics::Twist};
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
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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

        /// creates the adjoint 4x4 matrix for [Twist].
        /// ```rust
        /// use seigyo::{Matrix, statics::Twist};
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
        /// assert_eq!(twist.adjoint(), Matrix::from([
        ///     [0., 2., 0., 2.8],
        ///     [-2., 0., 0., 4.],
        ///     [0., 0., 0., 0.],
        ///     [0., 0., 0., 0.],
        /// ]));
        /// ```
        pub fn adjoint(&self) -> Matrix<4, 4, T> {
            let adjoint = self.angular.skew_symmetric();
            let zero = T::zero().into();

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
        /// use seigyo::{Matrix, statics::Twist};
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
        /// use seigyo::{Matrix, statics::Twist};
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
        /// use seigyo::{Matrix, statics::Twist};
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
        /// use seigyo::{Matrix, statics::Twist};
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

    impl<T: Float> Screw<T> {
        /// Creates a new screw from angular and linear matrices.
        ///
        /// Returns an error if the axis is not normalized:
        /// - For rotation screws (non-zero angular): the angular axis must be a unit vector.
        /// - For translation screws (zero angular): the linear axis must be a unit vector.
        ///
        /// ### Success: normalized rotation axis
        /// ```rust
        /// use seigyo::{Matrix, statics::Screw};
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
        /// use seigyo::{Matrix, statics::{Screw, Error, ScrewError}};
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
        /// use seigyo::{Matrix, statics::{Screw, Error, ScrewError}};
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
        pub fn new(angular: Matrix<3, 1, T>, linear: Matrix<3, 1, T>) -> Result<Self> {
            // linear component only needs to be normal when the Screw represents a translation motion (angular component is zero).
            let is_translation = angular.is_zero();
            let (axis_to_check, error) = if is_translation {
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

        /// Multiplying a [`Screw`] by theta scales its angular and linear components, producing the corresponding [`Twist`].
        /// ```rust
        /// use seigyo::{Matrix, statics::{Screw, Twist}};
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
        /// use seigyo::{Matrix, statics::{Screw, Twist}};
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
        /// ```        #[inline]
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
            let one_minus_cos = Complex::from(T::one() - theta.cos());
            let sin = theta.sin();

            // 2. Rodrigues' formula
            let rotation = Matrix::new_identity() + Complex::from(sin) * skew.clone() + one_minus_cos * skew_sq.clone();

            // 3. Translation matrix
            let translation = (Matrix::new_identity() * theta + one_minus_cos * skew + Complex::from(theta - sin) * skew_sq) * self.linear;

            Transformation { rotation, translation }
        }
    }

    impl<T: Float> TryFrom<Matrix<6, 1, T>> for Screw<T> {
        type Error = Error;

        fn try_from(value: Matrix<6, 1, T>) -> Result<Self> {
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

    impl<T: Float> core::ops::Mul<T> for Screw<T> {
        type Output = Twist<T>;

        fn mul(self, rhs: T) -> Self::Output {
            self.to_twist(rhs)
        }
    }

    /// Trait implements forward and backward kinematics for manipulator arms.
    pub trait Kinematics<const N: usize> {
        /// **Reference:** Example 4.7 from Modern Robotics
        /// ```rust
        /// use seigyo::{Matrix, statics::{Transformation, Screw, Kinematics}};
        /// use core::f32::consts::PI;
        ///
        /// struct Manipulator;
        ///
        /// impl Kinematics<7> for Manipulator{}
        ///
        /// let m = Transformation::try_from(Matrix::from([
        ///     [1., 0., 0., 0.],
        ///     [0., 1., 0., 0.],
        ///     [0., 0., 1., 0.060 + 0.3 + 0.55],
        ///     [0., 0., 0., 1.],
        /// ])).unwrap();
        ///
        /// let b_list = [
        ///     (Screw::new(Matrix::from([[0.], [0.], [1.]]), Matrix::new_zero()).unwrap(), 0.),
        ///     (Screw::new(Matrix::from([[0.], [1.], [0.]]), Matrix::from([[0.060 + 0.3 + 0.55], [0.], [0.]])).unwrap(), PI / 4.),
        ///     (Screw::new(Matrix::from([[0.], [0.], [1.]]), Matrix::new_zero()).unwrap(), 0.),
        ///     (Screw::new(Matrix::from([[0.], [1.], [0.]]), Matrix::from([[0.060 + 0.3], [0.], [0.045]])).unwrap(), -PI / 4.),
        ///     (Screw::new(Matrix::from([[0.], [0.], [1.]]), Matrix::from([[0.], [0.], [0.]])).unwrap(), 0.),
        ///     (Screw::new(Matrix::from([[0.], [1.], [0.]]), Matrix::from([[0.060], [0.], [0.]])).unwrap(), -PI / 2.),
        ///     (Screw::new(Matrix::from([[0.], [0.], [1.]]), Matrix::from([[0.], [0.], [0.]])).unwrap(), 0.),
        /// ];
        ///
        /// let fk = Manipulator::forward_in_body_frame(m, b_list);
        ///
        /// let fk_prediction = Matrix::from([
        ///     [0., 0., -0.99999994, 0.31572855],
        ///     [0., 1., 0., 0.],
        ///     [0.99999994, 0., 0., 0.65708894],
        ///     [0., 0., 0., 1.],
        /// ]);
        ///
        /// assert_eq!(fk, fk_prediction);
        /// ```
        fn forward_in_body_frame<T: Float>(m: Transformation<T>, b_theta_list: [impl Into<Transformation<T>>; N]) -> Transformation<T> {
            b_theta_list.into_iter().fold(m, |acc, s| acc * s.into())
        }
        
        fn forward_in_space_frame<T: Float>(m: Transformation<T>, s_theta_list: [impl Into<Transformation<T>>; N]) -> Transformation<T> {
            s_theta_list.into_iter().rev().fold(m, |acc, s| s.into() * acc)
        }
    }

    // /// ## Fowrard kinematice in body frame
    // /// ### Example
    // pub fn forward_kinematice_in_body_frame<T: Float>(m: Transformation<T>, b_theta_list: &[(Screw<T>, T)]) -> Transformation<T> {
    //     b_theta_list.iter().fold(m, |acc, s| acc * Transformation::from(s))
    // }

    // /// ## Fowrard kinematice in space frame
    // /// ### Example
    // /// **Reference:** Example 4.5 from Modern Robotics
    // /// ```rust
    // /// use seigyo::{Matrix, statics::{Transformation, Screw, forward_kinematice_in_space_frame}};
    // /// use core::f32::consts::PI;
    // ///
    // /// let m = Transformation::try_from(Matrix::from([
    // ///     [-1., 0., 0., 0.425 + 0.392],
    // ///     [0., 0., 1., 0.109 + 0.082],
    // ///     [0., 1., 0., 0.089 - 0.095],
    // ///     [0., 0., 0., 1.],
    // /// ])).unwrap();
    // ///
    // /// let s_list = [
    // ///     (Screw::new(Matrix::from([[0.], [0.], [1.]]), Matrix::new_zero()).unwrap(), 0.),
    // ///     (Screw::new(Matrix::from([[0.], [1.], [0.]]), Matrix::from([[-0.089], [0.], [0.]])).unwrap(), -PI / 2.),
    // ///     (Screw::new(Matrix::from([[0.], [1.], [0.]]), Matrix::from([[-0.089], [0.], [0.425]])).unwrap(), 0.),
    // ///     (Screw::new(Matrix::from([[0.], [1.], [0.]]), Matrix::from([[-0.089], [0.], [0.425 + 0.392]])).unwrap(), 0.),
    // ///     (Screw::new(Matrix::from([[0.], [0.], [-1.]]), Matrix::from([[-0.109], [0.425 + 0.392], [0.]])).unwrap(), PI / 2.),
    // ///     (Screw::new(Matrix::from([[0.], [1.], [0.]]), Matrix::from([[0.095 - 0.089], [0.], [0.425 + 0.392]])).unwrap(), 0.),
    // /// ];
    // ///
    // /// let fk = forward_kinematice_in_space_frame(m, &s_list);
    // ///
    // /// let fk_prediction = Matrix::from([
    // ///     [0., -1., 0., 0.095],
    // ///     [1., 0., 0., 0.10899997],
    // ///     [0., 0., 1., 0.98800004],
    // ///     [0., 0., 0., 1.],
    // /// ]);
    // ///
    // /// assert_eq!(fk, fk_prediction);
    // /// ```
    // pub fn forward_kinematice_in_space_frame<T: Float>(m: Transformation<T>, s_theta_list: &[(Screw<T>, T)]) -> Transformation<T> {
    //     s_theta_list.iter().rev().fold(m, |acc, s| Transformation::from(s) * acc)
    // }

    // /// Method to find the space jacobian
    // pub fn jacobian_space<T: Float, const N: usize>(s_theta_list: &[(Screw<T>, T); N]) -> Matrix<6, N, T> {
    //     s_theta_list.iter().enumerate().fold((Matrix::new_zero(), Transformation::<T>::default()), |(mut jacobian, mut transformation), (i, s_theta)| {
    //         let c: Matrix<6, 1, T> = (&transformation * &s_theta.0).into();
    //         (0..6).for_each(|r| jacobian[r][i] = c[r][0]);
    //         transformation *= Transformation::from(s_theta);
    //         (jacobian, transformation)
    //     }).0
    // }

    // /// Method to determine the body jacobian
    // pub fn jacobian_body<T: Float, const N: usize>(b_theta_list: [(Screw<T>, T); N]) -> Matrix<6, N, T> {
    //     todo!()
    // }


    #[derive(Debug, PartialEq)]
    pub enum Error {
        /// The provided rotation matrix is singular
        RotationSingular,
        /// The provided rotation matrix doesn't have orthogonal columns
        RotationNonOrthogonal,
        /// The rotation matrix has invalid orientation
        RotationInvalidOrientation,
        Screw(ScrewError),
    }

    impl Display for Error {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            match self {
                Self::RotationSingular => write!(f, "singular matrix received while creating rotation matrix"),
                Self::RotationNonOrthogonal => write!(f, "columns of rotation matrix are not orthogonal"),
                Self::RotationInvalidOrientation => write!(f, "rotation matrix not right handed. this represents reflection matrix."),
                Self::Screw(e) => write!(f, "error while making transformation matrix. {e}"),
            }
        }
    }

    impl From<crate::MatrixError> for Error {
        fn from(value: crate::MatrixError) -> Self {
            match value {
                crate::MatrixError::Singular => Self::RotationSingular,
            }
        }
    }

    impl From<ScrewError> for Error {
        fn from(value: ScrewError) -> Self {
            Self::Screw(value)
        }
    }

    #[derive(Debug, PartialEq)]
    pub enum ScrewError {
        RotationAxisNotNormal,
        TranslationAxisNotNormal,
    }

    impl Display for ScrewError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::RotationAxisNotNormal => write!(f, "screw axis should be normalized"),
                Self::TranslationAxisNotNormal => write!(f, "translation vector should be normal if there is no rotation (theta defines the magnitude of translation).")
            }
        }
    }
}
