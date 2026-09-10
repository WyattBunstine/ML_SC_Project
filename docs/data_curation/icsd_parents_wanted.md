# ICSD parent structures wanted

Targets whose oxygen (or hydrogen) content exceeds what the best available
Materials Project parent can host on its anion sites. Each row is a family that
shares one parent; a single downloaded CIF unlocks all of its entries.

**Where to put them:** `database/datafiles/MP/ICSD_Parent_Cifs/`

**Naming:** keep the ICSD download default, `EntryWithCollCode<code>.cif`
(that is what `scripts/nemad_expansion/scan_icsd.py` globs for).

| chemical system | entries | max Tc (K) | example target | closest MP parent | extra O per cation |
|---|---|---|---|---|---|
| Co-H-Na-O | 23 | 4.7 | Na0.3Co1D3.6O3.8 | Na5CoHO4 | +3.60 |
| Ba-Cu-H-O-Y | 19 | 93.8 | Y1Ba2Cu3H5O7 | Ba4Y(CuO3)3 | +0.14 |
| Ca-Cu-O-Sr | 19 | 110.0 | Sr12.5Ca1.5Cu24O41 | Sr3Ca(CuO2)4 | +0.08 |
| Ba-Ca-Cu-La-O-Y | 17 | 83.0 | Y0.6Ca0.4Ba1.1La0.9Cu3O7.18 | Ba8CaY3(CuO2)12 | +0.19 |
| Ba-Cu-O-Sr-Y | 15 | 86.1 | Y1Ba0.74Sr1.26Cu3O6.98 | Ba2Sr2Y2Cu6O13 | +0.07 |
| Bi-Cu-La-O-Sr | 13 | 35.5 | Bi1.95Sr1.65La0.4Cu1O6.414 | Sr3LaCu2(BiO3)4 | +0.07 |
| Ba-Ca-Cu-Gd-La-O | 12 | 80.0 | La1Gd1Ba2Ca2Cu6O14.1 | Ba3CaLa2Cu6O13 | +0.10 |
| Ba-Cu-La-O-Y | 12 | 69.1 | Y2La2Ba2Cu6O14.45 | Ba3LaY2(Cu3O7)2 | +0.03 |
| Ba-Ca-Cu-La-Nd-O | 12 | 84.0 | Nd0.9Ca0.1Ba0.6La0.4Cu3O6.7 | Ba3CaLa2Cu6O13 | +0.19 |
| Bi-Cu-La-O-Pb-Sr | 12 | 28.4 | Bi1Pb1Sr1.26La0.26Cu1O6 | Sr2LaCu2(BiO4)2 | +0.07 |
| As-Ca-Co-Fe-H | 9 | 23.2 | Ca1Fe0.98Co0.02As1H1 | Ca(FeAs)2 | +0.33 |
| B-Ba-Cu-O-Sr-Y | 9 | 51.0 | Y1Sr1.85Ba0.15Cu2.5B0.5O7 | BaSrY(CuO2)4 | +0.02 |
| As-Ce-Co-Fe-H | 9 | 23.0 | Ce1Fe0.98Co0.02As1H1 | Ce(FeAs3)4 | +0.33 |
| Ca-Cl-Cu-O | 8 | 38.0 | Ca2Cu1Cl2.5O2 | Ca2Cu(ClO)2 | +0.08 |
| Ba-Co-Cu-O-Y | 8 | 70.0 | Y1Ba1.75Cu2.75Co0.25O7.06 | Ba2Y(CuO2)3 | +0.18 |
| Ba-Ca-Cu-O-Pb-Sr-Y | 7 | 77.0 | Pb0.5Sr1.8Ba0.2Y0.85Ca0.15Cu2.5O6.98 | Sr8CaY3Cu12(PbO4)8 | +0.19 |
| Ba-Cu-O-Pr-Y | 7 | 68.3 | Y0.8Pr0.2Ba2Cu3O7.67 | Ba10PrY4(Cu3O7)5 | +0.08 |
| Bi-Ca-Cu-O-Sr | 7 | 94.2 | Bi8Sr8Ca4Cu9O34.25 | Sr2CaCu2(BiO4)2 | +0.04 |
| Ba-Cu-O-Sm | 7 | 31.8 | Sm1.6Ba1.4Cu3O7.11 | Ba2Sm(CuO2)4 | +0.04 |
| Ba-Ca-Cu-La-O-Yb | 6 | 76.0 | La1.1Yb0.7Ba1.1Ca1.4Cu5O11.692 | Ba3CaLa2Cu6O13 | +0.15 |
| Ba-Ca-Cu-La-O | 6 | 76.0 | La2Ca2Ba2Cu3O13.712 | Ba3CaLa2Cu6O13 | +0.07 |
| Ba-Cu-O-Sm | 6 | 93.5 | Sm1.28Ba1.72Cu3O7.03 | Ba10Sm5(Cu5O11)3 | +0.07 |
| Ba-Cu-H-O-Y | 6 | 94.5 | Y1Ba2Cu3H0.21O7 | Ba2Y(CuO2)4 | +0.05 |
| Ba-Cu-Fe-O-Y | 6 | 85.0 | Y1Ba2Cu2.7Fe0.3O7.15 | Ba2Y(CuO2)3 | +0.18 |
| Cu-La-O-Pb | 6 | 33.4 | Pb2Cu0.9La1.1Cu2O8.05 | LaCuPb | +1.14 |

Total: 261 entries over 25 families (>=6 entries each).
A further ~480 oxygen-excess entries sit in families of 1-5 rows.

## Priority

1. **Y-123 with excess oxygen / hydrogen** (`Ba-Cu-H-O-Y`, `Ba-Ca-Cu-La-O-Y`,
   `Ba-Cu-O-Sr-Y`, `Ba-Cu-La-O-Y`): ~70 entries, most above 60 K. One
   oxygen-loaded 123 structure (O7+delta with the interstitial site) covers several.
2. **Sr-Ca cuprate ladders** (`Ca-Cu-O-Sr`, 19 entries, up to 110 K):
   the Sr14-xCaxCu24O41 ladder phase.
3. **Bi-2201/2212 with excess O** (`Bi-Cu-La-O-Sr`, `Bi-Cu-La-O-Pb-Sr`,
   `Bi-Ca-Cu-O-Sr`): ~32 entries; a Bi-2201 with interstitial oxygen.
4. **Sodium cobaltate hydrate** (`Co-H-Na-O`, 23 entries): NaxCoO2 . yH2O.
   Needs the hydrated structure, not the anhydrous MP phase.
5. **1111 arsenide hydrides** (`As-Ca-Co-Fe-H`, `As-Ce-Co-Fe-H`, 18 entries):
   the hydrogen-substituted 1111 (LaFeAsO1-xHx type).
