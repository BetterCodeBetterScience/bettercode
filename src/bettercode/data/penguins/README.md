# Palmer Penguins Data

This directory contains Palmer Penguins data downloaded from the Palmer Station Antarctica LTER data repository.

## Data Sources

The data were downloaded from the Environmental Data Initiative (EDI) repository:

- **Adelie penguins**: `knb-lter-pal.219.5`
  - URL: http://pasta.lternet.edu/package/data/eml/knb-lter-pal/219/5/002f3893385f710df69eeebe893144ff
  - Title: Structural size measurements and isotopic signatures of foraging among adult male and female Adélie penguins (Pygoscelis adeliae) nesting along the Palmer Archipelago near Palmer Station, 2007-2009

- **Gentoo penguins**: `knb-lter-pal.220.7`
  - URL: http://pasta.lternet.edu/package/data/eml/knb-lter-pal/220/7/e03b43c924f226486f2f0ab6709d2381
  - Title: Structural size measurements and isotopic signatures of foraging among adult male and female gentoo penguins (Pygoscelis papua) nesting along the Palmer Archipelago near Palmer Station, 2007-2009

- **Chinstrap penguins**: `knb-lter-pal.221.8`
  - URL: http://pasta.lternet.edu/package/data/eml/knb-lter-pal/221/8/fe853aa8f7a59aa84cdd3197619ef462
  - Title: Structural size measurements and isotopic signatures of foraging among adult male and female Chinstrap penguins (Pygoscelis antarcticus) nesting along the Palmer Archipelago near Palmer Station, 2007-2009

## Data Creators

- Palmer Station Antarctica LTER
- Kristen Gorman - Simon Fraser University, Vancouver, BC

## Contact

- PAL LTER Information Manager, Palmer Station Antarctica LTER - pallter.im@gmail.com

## How to Update

To download fresh data from the original sources:

```python
from bettercode.penguindata import load_all_species

# Force download from source URLs
df = load_all_species(force_download=True)
```

Or run the download script:

```bash
python scripts/download_penguin_data.py
```

## Citation

Gorman KB, Williams TD, Fraser WR (2014). Ecological sexual dimorphism and environmental variability within a community of Antarctic penguins (genus Pygoscelis). PLoS ONE 9(3):e90081. https://doi.org/10.1371/journal.pone.0090081
