import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('HAM10000_metadata_balanced.csv')

# Create age bins from 0 to max age with step 5
bins = np.arange(0, df['age'].max() + 5, 5)
age_binned = pd.cut(df['age'].dropna(), bins=bins, right=False)

age_counts = age_binned.value_counts(sort=False)

plt.figure(figsize=(12,6))
age_counts.plot.bar(color='skyblue', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age Range')
plt.ylabel('Count')

# Set x-axis labels to be more explicit
plt.xticks(
    ticks=np.arange(len(age_counts)),
    labels=[f'{int(interval.left)}-{int(interval.right-1)}' for interval in age_counts.index],
    rotation=45
)

plt.tight_layout()
plt.savefig('age_distribution_bar_explicit.png', dpi=300)
plt.close()

# Sex plot
plt.figure(figsize=(12,6))
df['sex'].value_counts().plot.bar(color='orange')
plt.title('Sex Distribution')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('sex_distribution.png', dpi=300)
plt.close()

# Skin region plot
plt.figure(figsize=(14,6))
df['localization'].value_counts().head(10).plot.bar(color='green')
plt.title('Top 10 Skin Regions')
plt.xlabel('Skin Region')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('skin_region_distribution.png', dpi=300)
plt.close()
