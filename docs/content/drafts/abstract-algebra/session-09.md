---
title: "Part 9: Quotient groups — making cosets into elements"
slug: "quotient-groups"
summary: "Treat each coset as one element, define an operation on cosets, and see why the result must not depend on representatives."
subject: "Mathematics"
section: "Abstract Algebra"
type: "Learning session"
status: "approved-for-implementation"
---

# Part 9: Quotient groups — making cosets into elements

Part 8 sent every even integer to $0$ and every odd integer to $1$. The map ignored which *particular* even or odd integer we chose. Can we build a group whose elements are those two whole cosets, rather than individual integers?

**By the end:** explain what the elements of a quotient group are, add cosets without depending on their representatives, and state why a normal subgroup is needed in the general construction.

## Start with the two familiar cosets

Work in $(\mathbb Z,+)$ and let $H=2\mathbb Z$. Part 6 gave us exactly two distinct cosets:

$$
E=0+H=H=\{\ldots,-4,-2,0,2,4,\ldots\},
\qquad
O=1+H=\{\ldots,-3,-1,1,3,5,\ldots\}.
$$

The proposed new set is $\mathbb Z/H=\{E,O\}$. Each **element** of this new set is itself a **set of integers**. In particular, $5$ is an integer in the parent group, while $5+H=O$ is one element of the quotient set. The slash in $\mathbb Z/H$ names the quotient construction; it does not mean ordinary numerical division.

Part 8's parity map assigns the label $0$ to $E$ and $1$ to $O$. These labels are useful, but the quotient elements are the cosets $E$ and $O$ themselves.

## How should cosets add?

Choose one integer from each coset, add those integers, and take the coset of the sum:

$$
(a+H)+(b+H)=(a+b)+H.
$$

For example, $3+H=O$ and $4+H=E$. Their sum is

$$
(3+H)+(4+H)=7+H=O.
$$

```interactive:quotient-cosets
```

The full operation has only four cases:

| $+$ | $E$ | $O$ |
| --- | --- | --- |
| $E$ | $E$ | $O$ |
| $O$ | $O$ | $E$ |

The last cell says $O+O=E$: odd plus odd is even. More explicitly,

$$
(1+H)+(1+H)=2+H=H=E.
$$

So the quotient has an identity $E$; $O$ is its own inverse. This two-element addition pattern matches addition modulo $2$ in Part 8.

## Does the answer depend on which representatives we choose?

We could name the odd coset as $1+H$, $3+H$, or $5+H$. If adding cosets gave different answers for these names, our proposed operation would not be a function. This is the **well-definedness** question.

Try two names for the same calculation:

$$
(1+H)+(1+H)=2+H=E,
\qquad
(5+H)+(3+H)=8+H=E.
$$

Both agree. To prove this for *all* choices, suppose $a+H=a'+H$ and $b+H=b'+H$. The coset test from Part 6 says $a-a'\in H$ and $b-b'\in H$. Since $H$ is a subgroup, their sum is in $H$:

$$
(a+b)-(a'+b')=(a-a')+(b-b')\in H.
$$

Therefore $(a+b)+H=(a'+b')+H$. The result depends on the cosets, not on the representative integers we happened to pick.

**Definition — quotient group in the additive case.** If $G$ is an abelian group and $H\le G$, then $G/H$ is the set of cosets $a+H$ with operation $(a+H)+(b+H)=(a+b)+H$. The coset $H$ is the identity, and $(-a)+H$ is the inverse of $a+H$. The representative argument above works because subtraction and addition respect $H$.

## Three cosets: what does 0, 1, 2 mean?

In $\mathbb Z/3\mathbb Z$, the three elements are the whole cosets $[0]=3\mathbb Z$, $[1]=1+3\mathbb Z$, and $[2]=2+3\mathbb Z$. The numbers $0,1,2$ are shorthand labels for those cosets. For example, $[1]=\{\ldots,-2,1,4,7,\ldots\}$.

We add the classes by adding representatives: $[2]+[2]=[4]=[1]$. Both $4$ and $1$ have remainder $1$ modulo $3$. The coset $[0]$ is included and is the quotient identity. Taking the quotient groups integers by their remainders and lets us calculate directly with these classes.

## What changes for a general group?

For a group $(G,\ast)$ that may not be commutative, a subgroup $N$ must be **normal** before its cosets form a quotient group. Normal means $gN=Ng$ for every $g\in G$; we write $N\trianglelefteq G$. Every subgroup of an abelian group is normal, so $2\mathbb Z$ caused no difficulty.

When $N\trianglelefteq G$, the quotient $G/N$ consists of the cosets $aN$, and the operation is

$$
(aN)\ast(bN)=(a\ast b)N.
$$

Why does normality help? If $a'=a\ast n_1$ and $b'=b\ast n_2$ for $n_1,n_2\in N$, then

$$
a'\ast b'=a\ast b\ast(b^{-1}\ast n_1\ast b)\ast n_2.
$$

Normality puts $b^{-1}\ast n_1\ast b$ in $N$. The remaining factors from $N$ combine into an element of $N$, so $a'\ast b'$ belongs to $(a\ast b)N$. Thus different representatives give the same product coset. Without normality, this representative-independent operation is not guaranteed.

The identity of $G/N$ is $N=eN$, and the inverse of $aN$ is $a^{-1}N$. Associativity comes from the operation in $G$. If $G$ is finite, the number of quotient elements is the number of cosets $[G:N]$ from Part 7.

## What did the quotient do?

The quotient groups together elements that differ by something in $N$. In $\mathbb Z/2\mathbb Z$, integers differing by an even number become the same quotient element. That is why $1+2\mathbb Z=5+2\mathbb Z$, even though $1\ne5$ as integers.

The parity map from Part 8 and this quotient describe the same even/odd grouping in two ways: the map gives each integer a value $0$ or $1$, while the quotient treats its entire coset as one element. We will study the general relationship between homomorphisms, kernels, and quotients later.

## Try it

1. List the distinct cosets in $\mathbb Z/3\mathbb Z$. Which coset contains $-1$?
2. In $\mathbb Z/3\mathbb Z$, compute $(2+3\mathbb Z)+(2+3\mathbb Z)$. Repeat using different representatives of the same two cosets.
3. Explain why $O=1+2\mathbb Z$ is an element of $\mathbb Z/2\mathbb Z$ even though $O$ is not a subgroup of $(\mathbb Z,+)$.
4. In the definition of addition on cosets, what would go wrong if two representatives of the same coset gave different results?

**Hint.** For question 1, use the remainders $0$, $1$, and $2$ modulo $3$. For question 2, $-1$ and $2$ are in the same coset because their difference is a multiple of $3$.
