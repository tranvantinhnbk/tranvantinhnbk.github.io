import json

# Example content from the <textarea>
textarea_content = """
class Solution:
    def maxArea(self, height: List[int]) -> int:
        res = 0
        left = 0
        right = len(height) - 1
        while left < right:
            currentArea = (right - left) * min(height[left], height[right])
            res = max(currentArea, res)
            direction = height[left] < height[right]
            left += direction
            right -= not direction
        return res
"""

# Serialize the content into JSON format
json_content = json.dumps({'content': textarea_content})

print(json_content)