import sys; sys.path.insert(0, '/scratch/gpfs/GROUP/USER/project/miles-q38-build')
from miles_team.rm import grade
print(grade('so x=3.\nAnswer: 204', '204'), grade('The answer is \\boxed{204}', '204'), grade('Answer: 17', '204'))
