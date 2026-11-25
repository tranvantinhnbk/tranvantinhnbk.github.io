---
title: "Part 6: Cosets — moving a subgroup through a group"
slug: "cosets-and-partitions"
summary: "Use a subgroup to sort its parent group into cosets, starting with the general definition and then integer remainders."
subject: "Mathematics"
section: "Abstract Algebra"
type: "Learning session"
status: "approved-for-implementation"
---

# Part 6: Cosets — moving a subgroup through a group

Part 5 started with one element and built the subgroup it generates. Now start with a subgroup $H$ of a group $G$ and ask: how can $H$ help us organize *all* of $G$?

**By the end:** form a coset from a chosen element of the parent group, recognize when two choices give the same coset, and explain why cosets divide the group into non-overlapping pieces.

## Define the objects before the example

Let $(G,\ast)$ be a group and let $H\le G$ be a subgroup. Choose an element $a\in G$. The element $a$ belongs to the **parent group** $G$; it does not have to belong to $H$.

**Definition — Left coset.** The left coset of $H$ determined by $a$ is

$$
aH=\{a\ast h\mid h\in H\}.
$$

Keep $a$ fixed and combine it on the left with **every** element $h$ of $H$. The result is a set of elements of $G$, not usually one element. Closure in $G$ ensures that every $a\ast h$ belongs to $G$.

The order matters in a group that may not be commutative: $aH$ puts $a$ on the left. We will use left cosets throughout this lesson. In an additive group, we write the same idea as

$$
a+H=\{a+h\mid h\in H\}.
$$

If $a\in H$, then $aH=H$. To see this, closure gives $aH\subseteq H$. For any $x\in H$, the element $a^{-1}\ast x$ also lies in $H$, and $a\ast(a^{-1}\ast x)=x$; hence $H\subseteq aH$. In particular, the identity coset is $eH=H$.

If $a\notin H$, the coset $aH$ is a shifted copy of $H$. It need not be a subgroup: $aH$ may not contain the identity.

## What does the construction tell us?

A coset lets us treat two elements of $G$ as belonging to the same piece when one can be reached from the other by an element of $H$. For left cosets, the exact test is

$$
aH=bH\quad\Longleftrightarrow\quad a^{-1}\ast b\in H.
$$

**Why?** If $aH=bH$, then $b=b\ast e\in bH=aH$, so $b=a\ast h$ for some $h\in H$. Multiplying by $a^{-1}$ on the left gives $a^{-1}\ast b=h\in H$.

Conversely, if $a^{-1}\ast b=h\in H$, then $b=a\ast h$. Therefore

$$
bH=(a\ast h)H=a(hH)=aH,
$$

because $hH=H$ whenever $h\in H$.

Different choices of $a$ can therefore name the **same** coset. We care about the resulting sets, not just the names used to describe them.

## Every element lands in one piece

Every $a\in G$ lies in $aH$, since $e\in H$ and $a=a\ast e$. So the cosets cover $G$.

Two left cosets are either equal or disjoint. Suppose $x$ belongs to both $aH$ and $bH$. Then $x=a\ast h_1=b\ast h_2$ for some $h_1,h_2\in H$. Rearranging in the parent group,

$$
a^{-1}\ast b=h_1\ast h_2^{-1}\in H.
$$

The equality test above now gives $aH=bH$. Thus an overlap means the two cosets were the same set all along. Together, **covering** and **equal-or-disjoint** mean that the left cosets *partition* $G$.

Each coset is also the same size as $H$ in the sense of a one-to-one correspondence: the map $h\mapsto a\ast h$ sends $H$ onto $aH$, and multiplying by $a^{-1}$ recovers $h$. For finite groups, this will let us count elements in a later lesson.

## Example: integer remainders are cosets

Now choose the parent group $(\mathbb Z,+)$ and its subgroup $H=3\mathbb Z$, the multiples of three.

The coset chosen by $0$ is

$$
0+H=H=\{\ldots,-6,-3,0,3,6,\ldots\}.
$$

Choosing $1$ or $2$ gives

$$
\begin{aligned}
1+H&=\{\ldots,-5,-2,1,4,7,\ldots\},\\
2+H&=\{\ldots,-4,-1,2,5,8,\ldots\}.
\end{aligned}
$$

These are precisely the integers with remainders $0$, $1$, and $2$ when divided by $3$. There are no other cosets: every integer $n$ can be written as $n=3q+s$ with $s\in\{0,1,2\}$, so $n+H=s+H$.

For example, $4+H=1+H$ because $4-1=3\in H$. The representatives $4$ and $1$ are different integers, but they name the same set. Meanwhile $1+H\ne2+H$ because $2-1=1\notin H$.

This is the familiar idea of “same remainder” expressed using a subgroup. The subgroup tells us which differences we agree to treat as belonging to the same piece.

## Example: the square rotations split in two

Take the square-rotation group from Parts 3–5,

$$
G=\{e,r,r^2,r^3\},\qquad r^4=e,
$$

and its subgroup $H=\{e,r^2\}$. Then

$$
\begin{aligned}
eH&=\{e,r^2\}=H,\\
rH&=\{r,r^3\}.
\end{aligned}
$$

These two cosets contain every rotation and do not overlap. Choosing $r^3$ instead of $r$ does not create a third coset: $r^3H=\{r^3,r^5\}=\{r^3,r\}=rH$.

Here $H$ and $rH$ each contain two elements. The parent group has four elements. This is a concrete preview of the counting question in the next lesson.

## Your turn

Stay in $(\mathbb Z,+)$, but choose $H=2\mathbb Z$.

1. Write out several elements of $0+H$ and $1+H$.
2. Why is $3+H=1+H$? Use the test involving a difference.
3. Can $0+H$ and $1+H$ share an integer? Explain.
4. Does the coset $1+H$ form a subgroup of $(\mathbb Z,+)$? Check whether it contains the identity.

**Hint.** A difference belongs to $2\mathbb Z$ exactly when it is even. The identity for addition is $0$.

## Next learning topic

For a **finite** group, equally sized cosets lead to Lagrange’s theorem: the size of a subgroup divides the size of its parent group. This is proposed for Part 7, not yet a completed lesson.
