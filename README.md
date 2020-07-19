# Variational Autoencoders

The following repository is dedicated to experimentation with Variational Autoencoders.

## Contents

1. Contents
1. Issues/Common pitfalls while training VAEs
1. References

## Issues/Common pitfalls while training VAEs

1. In case the input is shaped in a specific fashion ( unlike regular #batches x #features ), such as an image, be careful
while calculating the reconstruction loss. Correctly sum across all the dimensions corresponding to a sample.
