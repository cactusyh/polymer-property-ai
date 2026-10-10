# Full MD Physics Summary

Rows: 7,614
Usable MD polymers with at least one good/borderline physics target: 7,613

## Backbone Confidence
- `low`: 6,182
- `medium`: 1,388
- `high`: 44

## Target QC
| target | attempted | good | borderline | invalid | missing | good_percentage | usable_percentage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rg | 7613 | 4750 | 2215 | 648 | 1 | 62.4% | 91.5% |
| Rg_over_sqrt_Nbackbone | 7613 | 524 | 578 | 6511 | 1 | 6.9% | 14.5% |
| Re_over_Rg | 7613 | 823 | 482 | 6308 | 1 | 10.8% | 17.1% |
| q1 | 7613 | 6877 | 736 | 0 | 1 | 90.3% | 100.0% |
| q2 | 7613 | 2950 | 2107 | 2556 | 1 | 38.7% | 66.4% |
| q3 | 7613 | 2255 | 2371 | 2987 | 1 | 29.6% | 60.8% |
| relative_shape_anisotropy | 7613 | 2690 | 2450 | 2473 | 1 | 35.3% | 67.5% |
| asphericity | 7613 | 1899 | 1926 | 3788 | 1 | 24.9% | 50.2% |
| short_range_contact_fraction | 7613 | 1432 | 0 | 6181 | 1 | 18.8% | 18.8% |
| medium_range_contact_fraction | 7613 | 1431 | 0 | 6182 | 1 | 18.8% | 18.8% |
| long_range_contact_fraction | 7613 | 1430 | 0 | 6183 | 1 | 18.8% | 18.8% |
| torsion_entropy | 7613 | 1432 | 0 | 6181 | 1 | 18.8% | 18.8% |
| torsion_transition_rate | 7613 | 1432 | 0 | 6181 | 1 | 18.8% | 18.8% |

## Distributions
- `Rg`: {'count': 7613.0, 'mean': 28.752108334711444, 'std': 19.15874675270343, 'min': 9.792419826299108, '5%': 11.646780368789413, '25%': 13.911443363137376, '50%': 21.11270650073213, '75%': 38.56168013127115, '95%': 68.19554341438743, 'max': 145.54267299084552}
- `Rg_over_sqrt_Nbackbone`: {'count': 1432.0, 'mean': 1.2615186165977097, 'std': 0.750339347386432, 'min': 0.5257612920140429, '5%': 0.615128366119336, '25%': 0.7664236986851185, '50%': 0.987223380615319, '75%': 1.470275498623163, '95%': 2.9334018129595503, 'max': 6.521923124940289}
- `Re_over_Rg`: {'count': 1432.0, 'mean': 2.0528109975598845, 'std': 0.5002521003990198, 'min': 0.4842075545311554, '5%': 1.1373134000650333, '25%': 1.7313768209531557, '50%': 2.123188754247546, '75%': 2.4153810121105295, '95%': 2.7373856415991855, 'max': 3.614779350959949}
- `q1`: {'count': 7613.0, 'mean': 0.7217670687321911, 'std': 0.13959235379049345, 'min': 0.3943733178140481, '5%': 0.48992638418018386, '25%': 0.61216593607082, '50%': 0.722557861003019, '75%': 0.8390015929081724, '95%': 0.9356586776006206, 'max': 0.9940990268774852}
- `q2`: {'count': 7613.0, 'mean': 0.18796756498784709, 'std': 0.08635948416647894, 'min': 0.00431830443506848, '5%': 0.04893355804959143, '25%': 0.11895656412521359, '50%': 0.18969528577415398, '75%': 0.25677128398036775, '95%': 0.3256108290597614, 'max': 0.4008003999747209}
- `q3`: {'count': 7613.0, 'mean': 0.09026536627996183, 'std': 0.060045447128588815, 'min': 0.001582668687446385, '5%': 0.012262300664248861, '25%': 0.03672411310541188, '50%': 0.08203140264812019, '75%': 0.13520664483920686, '95%': 0.19983346398186858, 'max': 0.27407501255826944}
- `relative_shape_anisotropy`: {'count': 7613.0, 'mean': 0.40832622505520005, 'std': 0.23798170042366434, 'min': 0.016613259917145247, '5%': 0.07638450000320975, '25%': 0.20237654788207304, '50%': 0.3774584248994668, '75%': 0.5964798045443299, '95%': 0.8201615217371567, 'max': 0.9824178357256385}
- `asphericity`: {'count': 7613.0, 'mean': 977.7830050693785, 'std': 1585.8966289581824, 'min': 8.787318058659476, '5%': 34.77442444526895, '25%': 91.89095654925822, '50%': 303.18945548626436, '75%': 1148.218677447299, '95%': 4171.750679026323, 'max': 20449.92037394579}
- `short_range_contact_fraction`: {'count': 1432.0, 'mean': 0.021161592178770946, 'std': 0.01022013494674714, 'min': 0.0086, '5%': 0.011200000000000005, '25%': 0.015200000000000003, '50%': 0.018600000000000002, '75%': 0.022850000000000002, '95%': 0.045399999999999996, 'max': 0.0854}
- `medium_range_contact_fraction`: {'count': 1432.0, 'mean': 0.03491703910614525, 'std': 0.023150452891350533, 'min': 0.0, '5%': 0.009999999999999998, '25%': 0.018999999999999993, '50%': 0.03200000000000001, '75%': 0.041850000000000005, '95%': 0.08798000000000003, 'max': 0.19600000000000006}
- `long_range_contact_fraction`: {'count': 1432.0, 'mean': 0.06869413407821229, 'std': 0.02210688397684753, 'min': 0.0, '5%': 0.03342, '25%': 0.053399999999999996, '50%': 0.06759999999999998, '75%': 0.08304999999999998, '95%': 0.10509000000000002, 'max': 0.1632}
- `torsion_entropy`: {'count': 1432.0, 'mean': 0.9558132634464093, 'std': 0.09106844600706122, 'min': 0.16725103060858232, '5%': 0.8154795030481702, '25%': 0.8853277103108439, '50%': 0.9663626319468434, '75%': 1.0278644787457079, '95%': 1.0852330702381088, 'max': 1.0985661543080654}
- `torsion_transition_rate`: {'count': 1432.0, 'mean': 0.0005689260315683536, 'std': 0.00013357127285330546, 'min': 2.09375e-05, '5%': 0.000306875, '25%': 0.000513125, '50%': 0.0005771875, '75%': 0.000640625, '95%': 0.0007715312500000001, 'max': 0.0012278125}

## Compact Target Recommendation
Empirical correlation audit supports a compact default set while retaining all raw candidates in the table:

- `log_Rg`: usable 6,965/7,614; preferred over raw `Rg` for first multitask MD pretraining pass.
- `Re_over_Rg`: usable 1,305/7,614; keep for first multitask MD pretraining pass.
- `q1`: usable 7,613/7,614; keep for first multitask MD pretraining pass.
- `q2`: usable 5,057/7,614; keep for first multitask MD pretraining pass.
- `relative_shape_anisotropy`: usable 5,140/7,614; keep for first multitask MD pretraining pass.
- `short_range_contact_fraction`: usable 1,432/7,614; keep for first multitask MD pretraining pass.
- `medium_range_contact_fraction`: usable 1,431/7,614; keep for first multitask MD pretraining pass.
- `long_range_contact_fraction`: usable 1,430/7,614; keep for first multitask MD pretraining pass.
- `torsion_entropy`: usable 1,432/7,614; keep for first multitask MD pretraining pass.
- `log_torsion_transition_rate`: usable 1,432/7,614; preferred over raw transition rate for first multitask MD pretraining pass.

`lambda1_mean/lambda2_mean/lambda3_mean` are redundant with `Rg`; `q3` is redundant because `q1 + q2 + q3 = 1`; `asphericity` is strongly shape-related and should be retained only for ablation.
