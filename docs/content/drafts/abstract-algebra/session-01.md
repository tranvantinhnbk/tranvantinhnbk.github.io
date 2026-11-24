---
title: "Part 1: Sets, operations, and closure"
subject: "Mathematics"
section: "Abstract Algebra"
type: "Learning session"
status: "approved-for-implementation"
---

# Part 1: Sets, operations, and closure

Choose the objects. Define how to combine them. Then ask whether the result stays inside the same set.

**By the end.** You should be able to explain closure, prove a simple example, and disprove it with a counterexample.

## Sets: which objects?

A set specifies the objects under discussion. The integers form the set

$$
\mathbb{Z}=\{\ldots,-2,-1,0,1,2,\ldots\}.
$$

We write $3\in\mathbb{Z}$ to say “3 belongs to the integers,” while $\frac{1}{2}\notin\mathbb{Z}$ means that one half is not an integer.

The even integers form the subset

$$
E=\{2k\mid k\in\mathbb{Z}\}.
$$

Read this aloud: “E is the set of numbers 2k, where k is an integer.” Choosing $k=-1,0,1$ gives $-2,0,2$.

Every element of $E$ is an integer, so $E\subseteq\mathbb{Z}$. The symbol $\subseteq$ means “is a subset of.”

## Operations: how do we combine them?

**Definition — Binary operation.** A binary operation on a set $S$ assigns every ordered pair of elements of $S$ exactly one result in $S$:

$$
\ast:S\times S\longrightarrow S.
$$

Read $S\times S$ as “all ordered pairs from S.” Addition on $\mathbb{Z}$ sends the pair $(3,5)$ to $8$.

String concatenation combines “ab” and “cd” into “abcd”. An operation need not be ordinary arithmetic; the symbol $\ast$ does not necessarily mean multiplication.

## Closure: do we stay inside?

**Definition — Closure.** A chosen set is closed under an operation when combining any two of its elements produces another element of that same set:

$$
\forall a,b\in S,\quad a\ast b\in S.
$$

The symbol $\forall$ means “for every.” This is a claim about every pair of inputs, not just a few examples.

Closure is already required by the definition of a binary operation $S\times S\to S$. When testing a familiar rule on a chosen set, we must verify that it meets this requirement.

| Operation on $\mathbb{Z}$ | Closed? | Reason |
| --- | --- | --- |
| Addition | Yes | The sum is an integer. |
| Subtraction | Yes | The difference is an integer. |
| Multiplication | Yes | The product is an integer. |
| Ordinary division | No | $1/2$ lies outside $\mathbb{Z}$; division by zero is undefined. |

The set matters as much as the operation. Subtraction stays inside the integers, but can leave the nonnegative integers: $2-5=-3$.

## A proof for every pair of even integers

**Proposition.** The even integers are closed under addition.

**Proof.** Choose any two even integers. By the definition of evenness, write

$$
a=2m,\quad b=2n,\qquad m,n\in\mathbb{Z}.
$$

Add them, then factor out $2$:

$$
\begin{aligned}
a+b &= 2m+2n \\
    &= 2(m+n).
\end{aligned}
$$

Since $m+n$ is an integer, $2(m+n)$ is even. The result belongs to $E$, as required. $\square$

We did not choose special values for $m$ and $n$. The same reasoning applies to every pair of even integers. A few examples alone would not prove this universal claim on an infinite set.

## One counterexample is enough

Let $S=\{0,1,2\}$. Ordinary addition fails closure because $1+2=3$ is outside $S$.

Now keep the set but change the operation:

$$
a\ast b=(a+b)\bmod 3.
$$

Add the inputs, then take the remainder on division by $3$. For example, $1\ast2=0$ and $2\ast2=1$. Every remainder is $0$, $1$, or $2$, so the result always belongs to $S$.

We have kept the set but changed the operation. An algebraic structure records both: $(S,\ast)$.

## Your turn

Let

$$
O=\{2k+1\mid k\in\mathbb{Z}\}
$$

be the odd integers.

1. Are they closed under addition?
2. Are they closed under multiplication?

For each operation, give a general argument or one counterexample.

**Hint.** Write the inputs as $2m+1$ and $2n+1$, then simplify their sum and product. Does the result have the required form?

## Next learning topics

Associativity, identity, inverses, and groups are planned next; those lessons are not yet drafted.
