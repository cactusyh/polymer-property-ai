# Bulk MD Pilot Descriptor Report

## Coverage

- 48 original polymer-temperature trajectories: 12 polymers × 600/500/400/300 K.
- Fixed 200 ps production trajectories were analysed; 600 K has mixed 1 ps/10 ps stored frames, and finite lags use dump timestep rather than frame index.
- Interchain contacts use molecule IDs and heavy atoms. C2 is unavailable for all systems because no common chemically defensible backbone mapping was supplied.
- Potential energy is reported; intermolecular nonbonded energy and CED are explicitly unavailable because clean energy decomposition is absent.

## Descriptor availability

- STATIC_PACKING: density, fluctuation, specific volume, interchain contacts, coordination, and interchain RDF first-peak/first-shell summaries.
- MSD: 10/50/100 ps, unwrapped coordinates, chain-COM drift removal, and affine box-rescaling removal.
- LOCAL_DYNAMICS: generic torsion autocorrelation, three-state torsion persistence/transition rate, and interchain-contact persistence.

## Within-trajectory stability

The CSV contains four physical-time block means, SD, CV, and first-half/second-half difference for each static descriptor.  Later-ML suitability is interpreted jointly with independent-replicate results; no descriptor is approved based only on a fixed CV cutoff.

## Guardrails

No Tg estimate, density-intersection Tg, Tg model, or 150-polymer expansion was produced.
