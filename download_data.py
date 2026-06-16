import kagglehub

# Download latest version
path = kagglehub.competition_download('h-and-m-personalized-fashion-recommendations')

print("Path to competition files:", path)