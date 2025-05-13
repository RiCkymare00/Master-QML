# Andrea Sarli - ID 971707

## Table of Contents

1. [Exercise 1](#exercise-1)
2. [Exercise 2](#exercise-2)
3. [Exercise 3](#exercise-3)
4. [Exercise 4](#exercise-4)
5. [Exercise 5](#exercise-5)

## Exercise 1, slide 43

**Homework**  
- Verify that $\mathbb{R}^2(\mathbb{R})$ is a vector space, that is, it satisfies properties 1 to 8 of Definition 1.  
- Verify that $\|x\|_2$ is a norm.  
- Verify that the function $f(x) = a + bx$, with $a \in \mathbb{R} \setminus \{0\}$ and $b, x \in \mathbb{R}$, is not linear in $\mathbb{R}$.
### Solution

**1. Verifying that** $\mathbb{R}^2(\mathbb{R})$ **is a vector space:**

Given $\mathbb{R}^2$, that is, the set of ordered pairs $(x, y)$ with $x$ and $y$ in $\mathbb{R}$, is a vector space over $\mathbb{R}$. This means it is possible to define:

**Vector sum:** $(x_1, y_1) + (x_2, y_2) = (x_1 + x_2, y_1 + y_2)$  
**Scalar Multiplication** $\alpha \cdot (x, y) = (\alpha x, \alpha y)$

Now we verify that these operations satisfy the 8 axioms of a vector space over $\mathbb{R}$:

1. **Closure under addition:**  
   If $(x_1, y_1), (x_2, y_2) \in \mathbb{R}^2$, then $(x_1 + x_2, y_1 + y_2) \in \mathbb{R}^2$.
   By definition of $\mathbb{R}^2$, we know that $x_1, x_2, y_1, y_2 \in \mathbb{R}$.

    We define vector addition as:
    $$
    (x_1, y_1) + (x_2, y_2) = (x_1 + x_2, y_1 + y_2)
    $$

    Since $\mathbb{R}$ is closed under addition, it follows that $x_1 + x_2 \in \mathbb{R}$ and $y_1 + y_2 \in \mathbb{R}$.

    Therefore, $(x_1 + x_2, y_1 + y_2) \in \mathbb{R}^2$.

2. **Associativity of addition:**  
   $((x_1, y_1) + (x_2, y_2)) + (x_3, y_3) = (x_1 + x_2 + x_3, y_1 + y_2 + y_3) = (x_1, y_1) + ((x_2, y_2) + (x_3, y_3))$

3. **Existence of additive identity:**  
   There exists $(0, 0) \in \mathbb{R}^2$ such that $(x, y) + (0, 0) = (x, y)$ for all $(x, y) \in \mathbb{R}^2$.

4. **Existence of additive inverse:**  
   For every $(x, y) \in \mathbb{R}^2$, there exists $(-x, -y)$ such that $(x, y) + (-x, -y) = (0, 0)$.

5. **Commutativity of addition:**  
   $(x_1, y_1) + (x_2, y_2) = (x_2, y_2) + (x_1, y_1)$

6. **Compatibility of scalar multiplication with real multiplication:**  
   $(\alpha \beta) \cdot (x, y) = \alpha \cdot (\beta \cdot (x, y))$

7. **Multiplicative identity:**  
   $1 \cdot (x, y) = (x, y)$ for every $(x, y) \in \mathbb{R}^2$

8. **Distributivity:**  
   - $\alpha \cdot ((x_1, y_1) + (x_2, y_2)) = \alpha \cdot (x_1 + x_2, y_1 + y_2) = (\alpha x_1 + \alpha x_2, \alpha y_1 + \alpha y_2)$  
   - $(\alpha + \beta) \cdot (x, y) = (\alpha x + \beta x, \alpha y + \beta y)$

✅ Therefore, $\mathbb{R}^2$ is a vector space over $\mathbb{R}$.

---

**2. Verifying that** $\|x\|_2$ **is a norm:**

Given:  
$\|x\|_2 = \sqrt{x_1^2 +...+ x_n^2}$

This satisfies:
- Non-negativity and zero only for the zero vector since all terms are equal or grater than 0,
- Homogeneity: $\|\alpha x\| = |\alpha| \|x\|$

  **Proof:**  
  Let $x = (x_1, x_2, \dots, x_n) \in \mathbb{R}^n$ and let $\alpha \in \mathbb{R}$.  
  Then:
  $$
  \|\alpha x\|_2 = \sqrt{(\alpha x_1)^2 + (\alpha x_2)^2 + \dots + (\alpha x_n)^2} = \sqrt{\alpha^2(x_1^2 + x_2^2 + \dots + x_n^2)} = |\alpha| \sqrt{x_1^2 + x_2^2 + \dots + x_n^2} = |\alpha| \|x\|_2
  $$
- Triangle inequality: $\|x + y\| \leq \|x\| + \|y\|$

  **Proof:**  
  Let $x, y \in \mathbb{R}^n$. Then:
  $$
  \|x + y\|_2^2 = \sum_{i=1}^n (x_i + y_i)^2 = \sum_{i=1}^n (x_i^2 + 2x_i y_i + y_i^2) = \|x\|_2^2 + 2\langle x, y \rangle + \|y\|_2^2
  $$
  where $\langle x, y \rangle$ is the dot product.

  By the Cauchy-Schwarz inequality:  
  $|\langle x, y \rangle| \leq \|x\|_2 \cdot \|y\|_2$

  Therefore:  
  $$
  \|x + y\|_2^2 \leq \|x\|_2^2 + 2\|x\|_2\|y\|_2 + \|y\|_2^2 = (\|x\|_2 + \|y\|_2)^2
  $$
  Taking square roots on both sides:  
  $$
  \|x + y\|_2 \leq \|x\|_2 + \|y\|_2
  $$

Therefore, $\|x\|_2$ is a norm.

---

**3. Verifying that** $f(x) = a + bx$ **is not linear:**

A function $f: \mathbb{R} \to \mathbb{R}$ is linear if:  
$f(x + y) = f(x) + f(y)$  
$f(\alpha x) = \alpha f(x)$

For $f(x) = a + bx$, with $a \ne 0$:  
$f(x + y) = a + b(x + y) \ne f(x) + f(y) = 2a + bx + by$  
$f(\alpha x) = a + \alpha bx \ne \alpha(a + bx)$  
unless $a = 0$

Therefore, the function is **not linear**.


## Exercise 2, slide 65

The attendant should calculate the determinants of the following matrices:

1. $A = \begin{bmatrix} 2 & -1 \\ 3 & -3 \end{bmatrix}$

**Solution:**  
$\det(A) = (2)(-3) - (-1)(3) = -6 + 3 = -3$

2. $B = \begin{bmatrix} 2 & 1 & 2 \\ -1 & 0 & 9 \\ 3 & 1 & 3 \end{bmatrix}$

**Solution:**  
Expand along the first row:
$$
\det(B) = 2 \cdot \begin{vmatrix} 0 & 9 \\ 1 & 3 \end{vmatrix}
- 1 \cdot \begin{vmatrix} -1 & 9 \\ 3 & 3 \end{vmatrix}
+ 2 \cdot \begin{vmatrix} -1 & 0 \\ 3 & 1 \end{vmatrix}
$$

$$
= 2(0 \cdot 3 - 9 \cdot 1) - 1(-1 \cdot 3 - 9 \cdot 3) + 2(-1 \cdot 1 - 0 \cdot 3)
= 2(-9) - 1(-30) + 2(-1)
= -18 + 30 - 2 = 10
$$

3. $C = \begin{bmatrix} 1 & 2c \\ -3 & -3c \end{bmatrix}$, with $c \in \mathbb{R}$

**Solution:**  
$\det(C) = (1)(-3c) - (2c)(-3) = -3c + 6c = 3c$


---

## Exercise 3, slide 66

These are 4 vectors in $\mathbb{R}^3$.
The space $\mathbb{R}^3$ has dimension 3, so at most 3 linearly independent vectors can exist in it.
If there are more than 3 vectors, as in this case (4 vectors), they cannot be linearly independent.
Therfore the given vectors are not linearly independent.

---

**Q2: Compute** $I_3 \cdot I_3$

$I_3$ is the $3 \times 3$ identity matrix:

$$
I_3 = \begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

Multiplying $I_3$ by itself:
$$
I_3 \cdot I_3 = I_3
$$

✅ **Conclusion:** $I_3 \cdot I_3 = I_3$

## Exercise 4, slide 81

The attendant should solve the open Leontiev input-output model in which:  
$$
M = \begin{bmatrix}
0.10 & 0.15 & 0.12 \\
0.20 & 0.00 & 0.30 \\
0.25 & 0.10 & 0.30
\end{bmatrix}, \quad
d = \begin{bmatrix}
100 \\
200 \\
300
\end{bmatrix}
$$

The model requires solving the equation:  
$$
(I - M)p = d
$$

**Step 1: Compute** $I - M$
$$
I - M = \begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
-
\begin{bmatrix}
0.10 & 0.15 & 0.12 \\
0.20 & 0.00 & 0.30 \\
0.25 & 0.10 & 0.30
\end{bmatrix}
=
\begin{bmatrix}
0.90 & -0.15 & -0.12 \\
-0.2 & 1 & -0.30 \\
-0.25 & -0.1 & 0.70
\end{bmatrix}
$$

**Step 2: Solve the system** $(I - M)p = d$  
This is equivalent to solving the matrix equation:
$$
\begin{bmatrix}
0.90 & -0.15 & -0.12 \\
-0.2 & 1 & -0.30 \\
-0.25 & -0.1 & 0.70
\end{bmatrix}
\begin{bmatrix}
p_1 \\
p_2 \\
p_3
\end{bmatrix}
=
\begin{bmatrix}
100 \\
200 \\
300
\end{bmatrix}
$$


**Step 3: Compute the inverse of** \( I - M \) (numerically):
$$
(I - M)^{-1} \approx
\begin{bmatrix}
1.244 & 0.2173 & 0.3065 \\
0.399 & 1.1145 & 0.5461 \\
0.5015 & 0.2368 & 1.6160
\end{bmatrix}
$$

**Step 4: Multiply inverse with demand vector**  
$$
p = (I - M)^{-1} \cdot d \approx
\begin{bmatrix}
1.244 & 0.2173 & 0.3065 \\
0.399 & 1,11451 & 0.5461 \\
0.5015 & 0.2368 & 1.6160
\end{bmatrix}
\begin{bmatrix}
100 \\
200 \\
300
\end{bmatrix}
= 
\begin{bmatrix}
259.79 \\
426.63 \\
582.31
\end{bmatrix}
$$

✅ **Conclusion:** The production vector is  
$$
p = \begin{bmatrix}
259.79 \\
426.63 \\
582.31
\end{bmatrix}
$$


## Exercise 5, slide 96

1. Matrix $A = \begin{bmatrix} 3 & -4 \\ 2 & -6 \end{bmatrix}$

Determinant:
$$
\det(A) = (3)(-6) - (-4)(2) = -18 + 8 = -10 \neq 0
$$

Inverse:
$$
A^{-1} = \frac{1}{-10} \begin{bmatrix} -6 & 4 \\ -2 & 3 \end{bmatrix}
= \begin{bmatrix} 0.6 & -0.4 \\ 0.2 & -0.3 \end{bmatrix}
$$

⸻

2. Matrix $B = \begin{bmatrix} 2 & 2c \\ -3 & -3c \end{bmatrix}$ with $c \in \mathbb{R}$

Determinant:
$$
\det(B) = (2)(-3c) - (2c)(-3) = -6c + 6c = 0
$$

Since the determinant is zero, $B$ is not invertible.


⸻

Eigenvalue computation

Matrix A:

Characteristic polynomial:
$$
\det(A - \lambda I) = \begin{vmatrix} 3 - \lambda & -4 \\ 2 & -6 - \lambda \end{vmatrix}
= (3 - \lambda)(-6 - \lambda) + 8
= \lambda^2 + 3\lambda - 10
$$

Solve:
$$
\lambda = \frac{-3 \pm \sqrt{49}}{2} = \frac{-3 \pm 7}{2}
\Rightarrow \lambda_1 = 2,\quad \lambda_2 = -5
$$

Matrix B:

Given $B = \begin{bmatrix} 2 & 2c \\ -3 & -3c \end{bmatrix}$,
consider the characteristic polynomial:

$$
\begin{vmatrix}
2 - \lambda & 2c \\
-3 & -3c - \lambda
\end{vmatrix}
= (2 - \lambda)(-3c - \lambda) - (2c)(-3) = 0
$$

Expanding:
$$
-6c - 2\lambda + 3c\lambda + \lambda^2 + 6c = 0
$$

Simplifying:
$$
\lambda^2 + (3c - 2)\lambda = 0
$$

Factoring:
$$
\lambda(\lambda + 3c - 2) = 0
$$

So, the eigenvalues are:
$$
\lambda_1 = 0, \quad \lambda_2 = 2 - 3c
$$