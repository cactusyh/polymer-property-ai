# Frozen Bulk MD Tg Descriptor Schema

Schema frozen following DESCRIPTOR_GATE_PASS; it must not change based on Tg-model outcomes.

{
  "schema_name": "bulk_md_tg_descriptor_schema_frozen",
  "schema_version": "1.0.0",
  "status": "FROZEN_AFTER_12_POLYMER_DESCRIPTOR_GATE_PASS",
  "temperatures_K": [
    600,
    500,
    400,
    300
  ],
  "production_ps": 200,
  "primary_families": {
    "STATIC_PACKING": [
      "mean_density",
      "density_fluctuation",
      "specific_volume_cm3_g",
      "intermolecular_heavy_contact_fraction",
      "intermolecular_contact_count_per_atom",
      "heavy_coordination_mean",
      "heavy_coordination_std",
      "intermolecular_rdf_first_shell_coordination"
    ],
    "DYNAMICS": [
      "MSD_10ps_A2",
      "MSD_50ps_A2",
      "MSD_100ps_A2",
      "generic_torsion_autocorr_10ps",
      "generic_torsion_autocorr_50ps",
      "generic_torsion_autocorr_100ps",
      "generic_torsion_persistence_10ps",
      "generic_torsion_persistence_50ps",
      "generic_torsion_persistence_100ps",
      "generic_torsion_transition_rate_per_ps_10ps",
      "generic_torsion_transition_rate_per_ps_50ps",
      "generic_torsion_transition_rate_per_ps_100ps",
      "interchain_contact_persistence_10ps",
      "interchain_contact_persistence_50ps",
      "interchain_contact_persistence_100ps"
    ],
    "ENERGETICS": [
      "potential_energy_mean_kcal_mol",
      "potential_energy_per_atom_kcal_mol"
    ]
  },
  "temperature_response": [
    "values_at_600_500_400_300K",
    "slope_per_K",
    "600K_minus_300K",
    "500K_minus_300K",
    "400K_minus_300K"
  ],
  "excluded_unstable": [
    "intermolecular_rdf_first_peak_g"
  ],
  "explicitly_unavailable": {
    "C2": "No common chemically defensible backbone mapping across polymers.",
    "CED": "No rigorous energy decomposition.",
    "intermolecular_nonbonded_energy_decomposition": "No rigorous energy decomposition."
  },
  "freeze_rule": "Do not change this schema in response to Tg modeling results."
}
