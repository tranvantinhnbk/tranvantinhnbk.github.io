---
title: "Part 4: Subgroups — groups inside groups"
slug: "subgroups"
summary: "Recognize when a subset inherits a group structure, reuse associativity, and apply the subgroup test."
subject: "Mathematics"
section: "Abstract Algebra"
type: "Learning session"
status: "approved-for-implementation"
---

# Part 4: Subgroups — groups inside groups

You already know that the even integers form a group under addition. They also sit inside a larger group: the integers under addition.

$$
2\mathbb{Z}\subseteq\mathbb{Z}.
$$

This gives us a new question: when does a subset of a group remain a group using the same operation?

**By the end:** explain what a subgroup is, identify the properties inherited from the larger group, and prove or disprove that a subset is a subgroup.

## Keep the operation, restrict the set

**Definition — Subgroup.** Let $(G,\ast)$ be a group. A subset $H\subseteq G$ is a subgroup of $G$ if $H$ is itself a group under the operation inherited from $G$.

We write

$$
H\le G.
$$

Here $\le$ means “is a subgroup of,” not an ordering of numbers. The operation must be understood from context. We can write $(H,\ast)\le(G,\ast)$ when we want it explicit.

The operation is unchanged: the same two inputs must produce the same result as they did in $G$. We are only restricting which elements are available.

Every group has the subgroups $\{e\}$ and $G$ itself, where $e$ is its identity. So “subgroup” does not necessarily mean strictly smaller. A subgroup $H$ is called proper if $H\ne G$.

## We inherit associativity

Take any $a,b,c\in H$. Because $H\subseteq G$, these elements also belong to $G$. Therefore

$$
(a\ast b)\ast c=a\ast(b\ast c)
$$

already follows from associativity in $G$.

Once closure ensures that the operation stays inside $H$, this equality supplies associativity on $H$. We do not need to invent a new proof for each subset.

What might fail is whether the required elements remain available inside $H$.

## Three checks inside the chosen subset

**Subgroup criterion.** A subset $H$ of a group $G$ is a subgroup if:

1. The identity $e$ of $G$ belongs to $H$.
2. For every $a,b\in H$, the product $a\ast b$ belongs to $H$.
3. For every $a\in H$, its inverse in $G$ also belongs to $H$.

Together with inherited associativity, these give all the group requirements on $H$. In particular, the identity and inverses are the same elements as in the ambient group; we do not choose replacements to make a subset work.

For addition, the identity is $0$ and the inverse of $a$ is $-a$. For multiplication of nonzero rationals, the identity is $1$ and the inverse is $1/a$.

## Could the subset have a different identity?

Suppose $H$ were a group under the same operation but had identity $f$. Since $f\in H$, its identity rule gives

$$
f\ast f=f.
$$

We can calculate in the larger group $G$, where $f$ has an inverse. Multiply both sides on the left by $f^{-1}$:

$$
f^{-1}\ast(f\ast f)=f^{-1}\ast f.
$$

Associativity and the inverse rule in $G$ give

$$
(f^{-1}\ast f)\ast f=e,
$$

so $e\ast f=e$, hence $f=e$.

Thus a subgroup cannot choose a different identity. The calculation uses the inverse in $G$; it does not assume in advance that this inverse belongs to $H$. For addition, an identity $f$ would satisfy $f+f=f$, forcing $f=0$.

## Example: even integers inside the integers

Let $G=\mathbb{Z}$ with addition and $H=2\mathbb{Z}$.

- **Identity:** $0=2\cdot0\in H$.
- **Closure:** if $a=2m$ and $b=2n$, then $a+b=2(m+n)\in H$.
- **Inverses:** $-a=2(-m)\in H$.

Associativity comes from integer addition. Hence

$$
(2\mathbb{Z},+)\le(\mathbb{Z},+).
$$

This is the group from Parts 1–3, now viewed in relation to a larger group. Being a subgroup is a relationship between two groups with compatible operations.

## Counterexample: closure alone is not enough

Inside $(\mathbb{Z},+)$, consider $H=\{0,1,2,\ldots\}$.

This subset contains $0$, and adding two of its elements stays in the subset. Associativity is inherited. But $1$ has additive inverse $-1$ in $\mathbb{Z}$, and $-1\notin H$.

Thus this subset is not a subgroup. An inverse somewhere in the larger group is not enough; it must belong to the subset being tested.

## Example: keep only the half-turns

Recall the square-rotation group

$$
G=\{e,r,r^2,r^3\},\qquad r^4=e.
$$

Here $r$ is a counterclockwise quarter-turn, and the operation is composition. Consider

$$
H=\{e,r^2\}.
$$

It contains the identity. Its products stay inside:

| $\ast$ | $e$ | $r^2$ |
| --- | --- | --- |
| $e$ | $e$ | $r^2$ |
| $r^2$ | $r^2$ | $e$ |

In particular, $r^2\ast r^2=r^4=e$. Both elements are their own inverses, so $H\le G$.

By contrast, $\{e,r\}$ is not a subgroup: $r\ast r=r^2$ is missing. It fails closure.

## A shorter test

**Theorem — One-step subgroup test.** A nonempty subset $H$ of a group $G$ is a subgroup if and only if

$$
a,b\in H\quad\Longrightarrow\quad a\ast b^{-1}\in H.
$$

The expression uses the inverse in the larger group $G$. The test asks whether combining an element with another element's inverse stays inside $H$.

**Proof.** If $H$ is a subgroup, inverse closure gives $b^{-1}\in H$, and closure gives $a\ast b^{-1}\in H$.

Conversely, suppose $H$ is nonempty and satisfies the displayed condition.

- Choose $x\in H$. Taking $a=b=x$ gives $x\ast x^{-1}=e\in H$.
- For any $b\in H$, take $a=e$. Then $e\ast b^{-1}=b^{-1}\in H$.
- Now $b^{-1}\in H$, so applying the condition to $a$ and $b^{-1}$ gives $a\ast(b^{-1})^{-1}=a\ast b\in H$.

The inverse of $b^{-1}$ is $b$, since their products in either order are $e$. Thus identity, inverse closure, and closure all hold; associativity is inherited. $\square$

Nonemptiness matters: the condition about pairs would hold vacuously for the empty set, but the empty set has no identity and is not a group.

For additive groups the test becomes

$$
a,b\in H\quad\Longrightarrow\quad a-b\in H.
$$

This is a test involving subtraction; the group's operation is still addition. For nonzero rationals under multiplication, the expression is $a/b$ instead.

## Your turn

Inside the group of even integers under addition, consider the multiples of four:

$$
G=2\mathbb{Z},\qquad H=4\mathbb{Z}=\{4k\mid k\in\mathbb{Z}\}.
$$

1. Explain why $H\subseteq G$.
2. Show that $H$ is nonempty.
3. If $a=4m$ and $b=4n$, show that $a-b\in H$.
4. What does the subgroup test let you conclude? Is this subgroup proper?

**Hint.** Express $4k$ as twice an integer. For the subtraction test, factor out $4$. To decide whether the subgroup is proper, look for an even integer that is not a multiple of four.

An optional follow-up using Part 3: do the positive rational numbers form a subgroup of the nonzero rationals under multiplication? Use $a/b$ and remember the identity is now $1$.

## Next learning topic

Generated subgroups and cyclic groups: starting with an element and collecting everything its repeated operation produces. This is planned, not yet a completed lesson.
