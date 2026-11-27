---
title: "Part 8: Homomorphisms — preserving an operation"
slug: "homomorphisms"
summary: "Map one group into another and check whether combining before or after the map gives the same result."
subject: "Mathematics"
section: "Abstract Algebra"
type: "Learning session"
status: "approved-for-implementation"
---

# Part 8: Homomorphisms — preserving an operation

We have studied groups, subgroups, cosets, and a theorem that counts cosets. Now ask how two groups can be related by a map. If we combine two elements in the first group, does mapping the result agree with mapping each element first and combining in the second group?

**By the end:** state the homomorphism rule, trace both routes in a concrete example, and explain why checking a few pairs is useful but does not prove the rule for every pair.

## Two groups and a map

Let $G$ and $K$ be groups, and let $f:G\to K$ be a function. We will write $\ast$ for the operation in each group. The same symbol is convenient, but the operations can be different: the set containing the inputs tells us which operation to use.

**Definition — group homomorphism.** The map $f$ is a homomorphism if, for every $a,b\in G$,

$$
f(a\ast b)=f(a)\ast f(b).
$$

On the left, combine $a$ and $b$ in $G$, then map the result to $K$. On the right, map $a$ and $b$ separately, then combine their images in $K$. Both sides are elements of $K$, so we can ask whether they are equal.

## See the two sets and the three mappings

Use $G=(\mathbb Z,+)$ and $K=(\{0,1\},+\bmod 2)$. Define $f(n)=n\bmod 2$, taking the remainder in $\{0,1\}$. Here $\ast$ means ordinary addition in $G$ and addition modulo $2$ in $K$.

For $a=3$ and $b=4$, the source contains $a=3$, $b=4$, and $a\ast b=7$. Their mapped values are $f(a)=1$, $f(b)=0$, and $f(a\ast b)=1$. In the target, $f(a)\ast f(b)=1+0\bmod 2=1$. The two results agree.

The diagram below lets you choose other integers and follow the three mappings pair by pair.

```interactive:homomorphism
```

The diagram is an example, not a proof about every pair. Its inputs are limited to small integers for easy exploration; the mathematical claim concerns all integers.

## Why parity always preserves addition

For any integers $a$ and $b$, write $a=2q+r$ and $b=2s+t$, where $r,t\in\{0,1\}$ are their remainders modulo $2$. Then

$$
a+b=2(q+s)+(r+t).
$$

The even term $2(q+s)$ contributes remainder $0$. Therefore

$$
f(a+b)=(r+t)\bmod 2=f(a)+f(b)\bmod 2.
$$

This holds for every pair of integers, including negative integers. Hence parity is a homomorphism from $(\mathbb Z,+)$ to $(\{0,1\},+\bmod 2)$.

## A connection to cosets

Recall that $f(n)=n\bmod 2$ and that the identity in the target group $(\{0,1\},+\bmod 2)$ is $0$. Which inputs land on that identity?

Choose an integer below. The colored numbers and boxes show the two cosets; the selected number shows which target value and coset it reaches.

```interactive:parity-cosets
```

$$
\ker(f)=\{n\in\mathbb Z\mid f(n)=0\}
=\{\ldots,-4,-2,0,2,4,\ldots\}=2\mathbb Z.
$$

The **kernel** is the set of inputs mapped to the target identity. Here it is exactly the even integers. For example, $f(-4)=0$, $f(0)=0$, and $f(6)=0$. The kernel is a subgroup of $(\mathbb Z,+)$: it contains $0$, and adding or negating even integers keeps them even.

The other inputs map to $1$:

$$
\{n\in\mathbb Z\mid f(n)=1\}
=\{\ldots,-3,-1,1,3,5,\ldots\}=1+2\mathbb Z.
$$

This is the odd coset from Part 6. For example, $5+2\mathbb Z=1+2\mathbb Z$ because $5-1=4\in2\mathbb Z$. Similarly, $-3+2\mathbb Z=1+2\mathbb Z$ because $-3-1=-4\in2\mathbb Z$. Every odd integer differs from $1$ by an even integer, so all odd integers give that same coset.

Keep the two kinds of object distinct: $f(5)=1$ is **one element** of the target group, while $5+2\mathbb Z=1+2\mathbb Z$ is **a set of integers**. The map gives the same output to every integer in a coset: all evens go to $0$, and all odds go to $1$.

This hints at a quotient group. Its elements are the two cosets $2\mathbb Z$ and $1+2\mathbb Z$, and we can add cosets by adding representatives. For instance,

$$
(1+2\mathbb Z)+(1+2\mathbb Z)=2+2\mathbb Z=2\mathbb Z.
$$

That is the coset version of $1+1=0\pmod 2$. We will define quotient groups carefully in a later session.

## Try it

1. In the diagram, choose $a=-3$ and $b=4$. What are $f(a\ast b)$ and $f(a)\ast f(b)$?
2. Define $g:\mathbb Z\to\{0,1,2\}$ by $g(n)=n\bmod 3$, with addition modulo $3$ in the target. Does the same remainder argument show that $g$ is a homomorphism?
3. Suppose $h:G\to K$ agrees with the homomorphism rule for just one chosen pair $a,b$. Is that enough to call $h$ a homomorphism? Explain the quantifier in the definition.

**Hint.** In question 1, $-3+4=1$, and the remainder of $-3$ modulo $2$ is $1$.
