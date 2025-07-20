# LeetCode 解法パターン集

## 🎯 基本パターン

### 1. Two Pointers (双方向ポインタ)

**使用場面**:
- ソート済み配列での探索
- 回文判定
- 三数の和などの問題

**基本形**:
```python
def two_pointers(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        if condition:
            # 処理
            left += 1
        else:
            right -= 1
    return result
```

**典型問題**:
- Two Sum II (167)
- Valid Palindrome (125)
- Container With Most Water (11)

### 2. Sliding Window (スライディングウィンドウ)

**使用場面**:
- 部分配列/部分文字列の最適化問題
- 固定長または可変長の連続する要素

**基本形**:
```python
def sliding_window(arr, k):
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i-k] + arr[i]
        max_sum = max(max_sum, window_sum)
    
    return max_sum
```

**典型問題**:
- Longest Substring Without Repeating Characters (3)
- Minimum Window Substring (76)
- Sliding Window Maximum (239)

### 3. Binary Search (二分探索)

**使用場面**:
- ソート済み配列での探索
- 条件を満たす境界値の探索

**基本形**:
```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

**典型問題**:
- Binary Search (704)
- Find First and Last Position of Element (34)
- Search in Rotated Sorted Array (33)

## 🌳 木・グラフパターン

### 4. DFS (深さ優先探索)

**基本形**:
```python
def dfs(node):
    if not node:
        return
    
    # 前処理
    visited.add(node)
    
    # 子ノードを探索
    for child in node.children:
        if child not in visited:
            dfs(child)
    
    # 後処理
```

### 5. BFS (幅優先探索)

**基本形**:
```python
from collections import deque

def bfs(start):
    queue = deque([start])
    visited = set([start])
    
    while queue:
        node = queue.popleft()
        # 処理
        
        for neighbor in node.neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

## 🔄 動的プログラミング

### 6. Dynamic Programming (DP)

**1次元DP**:
```python
def dp_1d(arr):
    n = len(arr)
    dp = [0] * n
    dp[0] = arr[0]
    
    for i in range(1, n):
        dp[i] = max(dp[i-1], arr[i])  # 例
    
    return dp[n-1]
```

**2次元DP**:
```python
def dp_2d(matrix):
    m, n = len(matrix), len(matrix[0])
    dp = [[0] * n for _ in range(m)]
    
    for i in range(m):
        for j in range(n):
            dp[i][j] = # 状態遷移式
    
    return dp[m-1][n-1]
```

## 🔙 バックトラッキング

### 7. Backtracking

**基本形**:
```python
def backtrack(path, choices):
    if is_valid_solution(path):
        result.append(path[:])
        return
    
    for choice in choices:
        if is_valid_choice(choice):
            path.append(choice)
            backtrack(path, get_next_choices(choice))
            path.pop()  # バックトラック
```

**典型問題**:
- Permutations (46)
- Combination Sum (39)
- N-Queens (51)

## 📊 その他の重要パターン

### 8. Greedy (貪欲法)

**基本的な考え方**:
- 各ステップで局所的に最適な選択
- 全体として最適解が得られることを証明必要

### 9. Union-Find (素集合データ構造)

**基本形**:
```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
```

### 10. Trie (トライ木)

**基本形**:
```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
```

---

## 💡 パターン選択のヒント

1. **配列/文字列**: Two Pointers, Sliding Window
2. **探索**: Binary Search, DFS, BFS
3. **最適化**: DP, Greedy
4. **組み合わせ**: Backtracking
5. **グラフ**: DFS, BFS, Union-Find
6. **文字列**: Trie, KMP

問題を見たときに、まずどのカテゴリに属するかを考え、適切なパターンを選択しましょう。
