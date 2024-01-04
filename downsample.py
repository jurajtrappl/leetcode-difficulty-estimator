import numpy as np
from sklearn.model_selection import train_test_split

# 1. Load the dataset.
with open('leetcode_problems_dataset.json', 'r') as f:
    problems = json.load(f)
    
# 2. Remove HTML tags.
problems_without_html = []
for problem_name, problem_data in tqdm(problems.items()):
    if not problem_data["content"]:
        continue
    
    problems_without_html.append((BeautifulSoup(problem_data["content"], "html.parser").get_text(), problem_data["difficulty"]))
    
# 3. Create X, y
X, y = [], []
difficulties_int = { "Easy": 0, "Medium": 1, "Hard": 2 }
for problem_description, difficulty in problems_without_html:
    X.append(problem_description)
    y.append(difficulties_int[difficulty])
    
# 4. Prepare downsampling function.
def downsample_dataset(features, labels):
    """
    Downsamples the dataset to have an equal distribution of classes.

    Parameters:
    X (list): Feature data.
    y (list): Corresponding labels.

    Returns:
    tuple: Downsampled feature data and labels.
    """
    paired_data = list(zip(features, labels))
    class_distribution = Counter(labels)

    min_samples = min(class_distribution.values())

    downsampled_data = []
    class_counts = {cls: 0 for cls in class_distribution.keys()}

    for data, label in paired_data:
        if class_counts[label] < min_samples:
            downsampled_data.append((data, label))
            class_counts[label] += 1

    features_downsampled, labels_downsampled = zip(*downsampled_data)

    return list(features_downsampled), list(labels_downsampled)

# 5. Downsample dataset.
X, y = downsample_dataset(X, y)

# 6. (Optional)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=SEED, stratify=y_val)

X_train, X_val, X_test, y_train, y_val, y_test = np.array(X_train), np.array(X_val), np.array(X_test), np.array(y_train), np.array(y_val), np.array(y_test)
