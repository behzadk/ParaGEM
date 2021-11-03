# yeast_LAB_coculture
## Models
Lactococcus lactis              SB-261                          (Kefir paper)
Lactococcus lactis              SB-17                           (Kefir paper)
Lactococcus lactis              SB-352                          (Kefir paper)
Lactococcus lactis              253_2013_5140_MOESM6_ESM.xml    (Flahaut et al. 2013)
Lactococcus lactis              L_lactis.xml                    (Carveme paper)

Saccharomyces cerevisiae iAZ900        12918_2010_589_MOESM3_ESM.XML   (Ali R Zomorrodi & Costas D Maranas 2010)
Saccharomyces cerevisiae iTO977        12918_2012_1082_MOESM2_ESM.xml (Osterlund et al. 2013)
Saccharomyces cerevisiae iAZ900        iAZ900.xml                      https://github.com/bheavner/yeast_models/blob/master/iAZ900/iAZ900.xml

Lactobacillus plantarum         L_plantarum.XML                 (Carveme paper)

`smetana ./models/L_lactis/L_lactis.xml ./models/L_plantarum/L_plantarum.xml -d -o ./carveme_output/lactis_plant_ecol_ --solver 'gurobi' --flavor cobra`
`smetana ./models/L_lactis/L_lactis.xml ./models/L_plantarum/L_plantarum.xml ./models/E_col/E_coli_IAI1.xml -d -o ./carveme_output/lactis_plant_ecol --solver 'gurobi' --flavor cobra`

`smetana ./models/L_lactis/L_lactis_fbc.xml ./models/S_cerevisiae/iMM904.xml -d -o ./carveme_output/lactis_cerevisiae --solver 'gurobi' --flavor fbc2`
`smetana ./models/L_lactis/L_lactis_fbc.xml -d -o ./carveme_output/lactis_only --solver 'gurobi' --flavor fbc2`


`smetana ./models/S_cerevisiae/iMM904.xml -d -o ./carveme_output/lactis_cerevisiae --solver 'gurobi' --flavor cobra`

Manually curated model of Lactococcus lactis IL1403 was updated
in order to reconcile model’s in silico growth with L. lactis IL1403 inability to growth in the CDM35 medium and its known auxotrophies
(Table S4).

Furthermore, biosynthesis of leucine, arginine, valine, glutamine, glutamate and isoleucine was blocked, as L. lactis IL1403 is known to be unable to synthesize
these metabolites

## Workflow
1. Generate complete environment reactions.
2. Read SMETANA output full output
3. Identify metabolites involved in crossfeeding
4. Identify metabolites involved in competition
5. Identify metabolites in env that are not in media, not produced 