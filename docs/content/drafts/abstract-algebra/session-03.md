---
title: "Part 3: Groups — putting the properties together"
slug: "groups-and-first-examples"
summary: "Classify familiar operations, recognize rotations as algebra, and derive cancellation from the group requirements."
subject: "Mathematics"
section: "Abstract Algebra"
type: "Learning session"
status: "draft-for-author-review"
---

# Part 3: Groups — putting the properties together

Part 1 introduced closure. Part 2 introduced associativity, identity, and inverses. Now we combine these properties into one definition and see what it lets us prove.

A group is one kind of algebraic structure. Its requirements determine when the group label and its theorems apply. Operations that fail some requirements can still be studied.

## The group definition

**Definition — Group.** A group is a set $G$ with a binary operation $\ast:G\times G\to G$ satisfying:

1. **Associativity:** $(a\ast b)\ast c=a\ast(b\ast c)$ for all $a,b,c\in G$.
2. **Identity:** there is an $e\in G$ with $e\ast a=a\ast e=a$ for every $a\in G$.
3. **Inverses:** for every $a\in G$, there is an element $a^{-1}\in G$ with $a\ast a^{-1}=a^{-1}\ast a=e$.

Closure is included in the requirement that the operation maps $G\times G$ into $G$. Check it explicitly when examining an example.

Commutativity is an additional property, not a group requirement. A group is called abelian when $a\ast b=b\ast a$ for every pair of elements.

## First example: even integers under addition

Let

$$
E=2\mathbb{Z}=\{\ldots,-4,-2,0,2,4,\ldots\}.
$$

Check each requirement:

- **Closure:** if $a=2m$ and $b=2n$, then $a+b=2(m+n)$ is even.
- **Associativity:** addition of even integers inherits associativity from integer addition.
- **Identity:** $0$ is even and $0+a=a+0=a$.
- **Inverses:** if $a=2m$, then $-a=-2m=2(-m)$ is also even, and $a+(-a)=0$.

Therefore $(E,+)$ is a group. It is also abelian because integer addition is commutative.

The inverse must stay inside the chosen set; finding an undoing element somewhere else is not enough.

## A near miss: nonnegative integers under addition

For $\mathbb{Z}_{\ge0}=\{0,1,2,\ldots\}$, addition is closed and associative, and $0$ is an identity. But $1$ has no additive inverse in this set: $-1$ is not nonnegative.

Thus $(\mathbb{Z}_{\ge0},+)$ is not a group. It still has useful algebraic structure; it simply fails this definition.

## Groups can consist of actions

Consider the rotations that carry a square onto itself. Write

$$
G=\{e,r,r^2,r^3\},
$$

where $e$ means no rotation and $r$ means a counterclockwise quarter-turn. The operation combines rotations by composition. Powers represent repeated composition: $r^2$ is a $180^\circ$ rotation, and $r^3$ is a $270^\circ$ rotation. Four quarter-turns give the identity:

$$
r^4=e.
$$

A half-turn followed by a quarter-turn gives a three-quarter-turn. Four quarter-turns return the square to its starting position.

- **Closure:** combining allowed rotations gives another allowed rotation.
- **Associativity:** composition of these transformations is associative; regrouping three successive rotations does not change the final action.
- **Identity:** $e$ changes nothing.
- **Inverses:** $e$ is its own inverse, $r$ and $r^3$ undo each other, and $r^2$ is its own inverse.

So these rotations form a group. Its elements are transformations of a square, rather than ordinary numbers. The same definition applies to both examples.

## What do the requirements let us prove?

**Proposition — Left cancellation.** In a group, if $a\ast b=a\ast c$, then $b=c$.

**Proof.** Combine both sides on the left with the inverse of $a$:

$$
a^{-1}\ast(a\ast b)=a^{-1}\ast(a\ast c).
$$

Use associativity to regroup, keeping the order unchanged:

$$
(a^{-1}\ast a)\ast b=(a^{-1}\ast a)\ast c.
$$

The inverse and identity properties give

$$
e\ast b=e\ast c,
$$

hence $b=c$. $\square$

We used an inverse to undo $a$, associativity to move the parentheses, and the identity to simplify the result. We did not use commutativity.

Cancellation is not automatically valid for every operation. Under integer multiplication, $0\cdot2=0\cdot3$, while $2\ne3$. The group proof cannot be applied: $0$ has no multiplicative inverse.

## Your turn

Consider the nonzero rational numbers under multiplication:

$$
\mathbb{Q}^{\times}=\{q\in\mathbb{Q}\mid q\ne0\}.
$$

1. Is the product of two elements always in this set?
2. Is multiplication associative?
3. What is the identity?
4. What is the inverse of a general nonzero rational number, and does it remain in the set?

**Hint.** Write a nonzero rational number as $a/b$, where $a$ and $b$ are nonzero integers. Find a rational number whose product with it is $1$.

## Next learning topic

Subgroups: when a smaller part of a group still forms a group under the same operation. This is planned, not yet a completed lesson.
