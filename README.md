# js_divergence
Calculates the Jensen-Shannon divergence between two probability distributions using Monte-Carlo. 


### Installation

Requires numpy.


### Usage

This function calculates the Jensen-Shannon divergence of two probability functions using Monte-Carlo. This is useful if the explicit form of one or both of the probability functions is unknown. 

To calculate the divergence, first sample N times from each distribution. Then, calculate the pdf of each vector for each of the distributions. This leads to 4 vectors. Let's say samples x come from distribution P and samples y come from distribution Q. The resulting vectors are: 

- `prob_x_in_p`: pdf of P at x (P.pdf(x))
- `prob_x_in_q`: pdf of Q at x (Q.pdf(x))
- `prob_y_in_p`: pdf of P at x (P.pdf(y))
- `prob_y_in_q`: pdf of Q at y (Q.pdf(y))

Then, run:

```
div = js_divergence(prob_x_in_p, prob_x_in_q,
                    prob_y_in_p, prob_y_in_q)
```
