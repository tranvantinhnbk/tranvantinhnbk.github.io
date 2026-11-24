---
title: "Part 2: Associativity, identity, and inverses"
slug: "associativity-identity-inverses"
summary: "Explore regrouping, doing nothing, and undoing an operation through concrete examples."
subject: "Mathematics"
section: "Abstract Algebra"
type: "Learning session"
status: "approved-for-implementation"
---

# Part 2: Associativity, identity, and inverses

Part 1 asked whether an operation keeps its results inside the chosen set. Now we introduce three more properties: can we regroup operations, find an element that does nothing, and undo an element?

These properties help describe and classify a set with an operation. An operation can satisfy some and fail others. All three, together with a binary operation on the set, are required when we call the structure a group. We will study that structure more fully in a later session; this lesson concentrates on the three properties themselves.

## Associativity: change the grouping

**Definition — Associativity.** An operation $\ast$ on a set $S$ is associative when

$$
(a\ast b)\ast c=a\ast(b\ast c)
$$

for every $a,b,c\in S$.

The order of the three inputs stays the same. Only the parentheses change.

For example, integer addition gives

$$
\begin{aligned}
(2+3)+4&=5+4=9,\\
2+(3+4)&=2+7=9.
\end{aligned}
$$

This illustrates the rule. Integer addition is associative generally; one numerical example alone would not prove a statement about every triple of integers.

Subtraction gives a counterexample:

$$
\begin{aligned}
(8-3)-2&=3,\\
8-(3-2)&=7.
\end{aligned}
$$

Because the results differ, subtraction is not associative. It is still closed on the integers: closure and associativity are different properties.

**Do not confuse grouping with order.** Associativity compares $(a\ast b)\ast c$ with $a\ast(b\ast c)$. Commutativity compares $a\ast b$ with $b\ast a$. We introduce commutativity here only to distinguish the two ideas.

## Identity: an element that does nothing

**Definition — Identity.** An identity element $e\in S$ satisfies

$$
e\ast a=a\ast e=a
$$

for every $a\in S$.

Both sides matter: putting the identity before or after any element leaves that element unchanged.

For addition on the integers, the identity is $0$:

$$
0+a=a+0=a.
$$

For multiplication on the integers, the identity is $1$:

$$
1\cdot a=a\cdot1=a.
$$

The same set can have different identity elements under different operations.

For strings under concatenation, the identity is the empty string. Joining it before or after “hello” leaves “hello”. An identity need not be a number.

## Inverses: undo an element

**Definition — Inverse.** In a set with an operation and identity $e$, an inverse of $a$ is an element $b$ in that same set such that

$$
a\ast b=b\ast a=e.
$$

Under addition, the inverse of $7$ is $-7$:

$$
7+(-7)=(-7)+7=0.
$$

Under multiplication, the inverse of $7$ would be $\frac17$:

$$
7\cdot\frac17=\frac17\cdot7=1.
$$

But $\frac17\notin\mathbb{Z}$. Thus $7$ has no multiplicative inverse within the integers, even though it does within the nonzero rational numbers.

Always specify the set and operation. The symbol $a^{-1}$ refers to an inverse under the chosen operation; it does not universally mean the fraction $1/a$. For additive inverses, we normally write $-a$.

## Three different questions

| Property | What we ask | Addition on the integers |
| --- | --- | --- |
| Associativity | Can we move the parentheses without changing the result? | Yes. |
| Identity | Is there an element that leaves every element unchanged on either side? | Yes: $0$. |
| Inverses | Can each element be combined with an element of the same set to reach the identity on both sides? | Yes: the inverse of $a$ is $-a$. |

We are learning properties of operations, rather than requiring every operation in abstract algebra to behave this way. A later lesson will use them to define and explore groups.

## Your turn

Let $E=\{2k\mid k\in\mathbb{Z}\}$ be the even integers, with addition. We proved closure in Part 1.

1. Is addition on $E$ associative? Explain how associativity of integer addition applies.
2. Which element is the additive identity, and does it belong to $E$?
3. If $a=2k$, what is its additive inverse? Show that this inverse also belongs to $E$.

**Hint.** For the last question, try expressing the inverse in the form $2m$ for an integer $m$.

As a further question, consider the nonnegative integers $\{0,1,2,\ldots\}$ under addition. Which of the three properties holds, and which needs closer attention?
