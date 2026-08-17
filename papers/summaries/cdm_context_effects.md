# Summary: Discovering Context Effects from Raw Choice Data

**Paper:** Discovering Context Effects from Raw Choice Data (CDM Paper)

## Core Insight
Real-world choices systematically violate the Independence of Irrelevant Alternatives (IIA) assumption
underlying standard MNL/softmax models. The CDM models each item as playing a dual role:
as a choice target and as a context item that influences other items' utilities.

## Key Ideas
- IIA failure: adding a decoy item changes choice probabilities in ways MNL cannot explain
- CDM: utility of item x in context z is v(x|{z}) = v(x) + u_{xz}
- Reparametrization: u_{xz} = v(x|{z}) - v(z) (softmax invariance to additive constants)
- Low-Rank CDM: factorizes context matrix for scalability
- Identifiability: requires choice sets of mixed sizes; size-2 and size-n sets alone are insufficient

## Open Questions
- How does CDM performance degrade when choice set size distribution is skewed?
- Can CDM context effects be incorporated into LTR frameworks like LambdaRank?
