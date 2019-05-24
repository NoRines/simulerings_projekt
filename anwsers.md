

1. Done

2. Done

3. Done

4. The ns3 UniformRandomVariable does not seem to repeat and uses more values between 0 and 1. Using our implementation with the values seed=1, a=13, c=1 and m=100 creates a cycle with 20 values before repetition. Both are however uniformally distributed. Using a m that is a large prime would mean that we get as many values as possible between 0 and the large prime this is because p % n != 0 for any prime p and value n < p. Howerer if a and p are coprime then this also results in a simmilar way to using primes as m. So if we want to make the modulo operation as effective as possible we can choose a m that is a power of 2. Then if we choose an odd value as a, a and m are coprime.

5. Combined Multiple-Recursive Generator. The normal random variable in ns3 uses the polar form of the Box-Muller method which is a rejection sampling method that aviods using trigometric functions. https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform

6. Our implementation is faster we get a time of 3.320s for ns3 it takes 3.806s. We tested with 10 million generated values. With 1 billion generated values our takes ~18.468s while ns3 takes ~1m 7.544s

7. Acording to the lectures we can use inverse cdf to function on the generated uniform random variable to create a random variable following the desired distriobution. For exponential distrobution we can use the function inv(Fx(u)) = -(1/b) * ln(1-u).

8. When comparing our implementation with the ns3 implementation we get quite simmilar distributions. However our implementation seems have a bit less variance than the ns3 version.
