### Quantum Instanton

This code uses quantum instanton to calculate the scattering rate of a quantum particle over a 1D Eckart barrier. The quantum instanton rate is given by

$$k_{\text{QI}}=\frac{\hbar\sqrt{\pi}}{2}\frac{C_{dd}(0)}{Q_r}\frac{C_{ff}(0)}{C_{dd}(0)}\frac{1}{\Delta H}\,.$$

Usually the first term $\frac{C_{dd}(0)}{Q_r}$ is sampled by umbrella sampling, and the last two terms are sampled by Path Integral Monte Carlo. However, converging Monte Carlo for ring polymers are notoriously slow due to the extremely high dimensionality of the extended phase space. Therefore I do everything using constained MD with SHAKE and RATTLE algorithm. In addition, the first term is sampled by thermodynamic integration (because I like it), and I used a virial estimator instead of a traditional thermodynamic one, which gives much much lower variance. The most uncertainty comes to the $\frac{C_{ff}(0)}{C_{dd}(0)}$ sampling because I couldn't find a better estimator other than the usual thermodynamic one by direct differentiation.

### References:
 - Miller, W. H., Zhao, Y., Ceotto, M., & Yang, S. (2003). Quantum instanton approximation for thermal rate constants of chemical reactions. The Journal of chemical physics, 119(3), 1329-1342.
 - Yamamoto, Takeshi, and William H. Miller. "On the efficient path integral evaluation of thermal rate constants within the quantum instanton approximation." The Journal of chemical physics 120.7 (2004): 3086-3099.
