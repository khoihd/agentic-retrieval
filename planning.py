from collections import defaultdict, deque

from pydantic import BaseModel


class Task(BaseModel):
    id: int
    task: str
    dependencies: list[int]

def build_execution_order(plan: list[Task]):
    # plan[a] -> dependency = [b, c]
    # b -> a
    # c -> a

    task_ids = {task.id for task in plan}
    indegree = defaultdict(int)  # id -> number of dependencies
    nexttask_ids = defaultdict(set)  # id -> set of the next task

    for task in plan:
        if len(task.dependencies) != len(set(task.dependencies)):
            raise ValueError(f"Duplicate dependencies in task {task.id}")

        indegree[task.id] = len(task.dependencies)
        for dependent in task.dependencies:
            nexttask_ids[dependent].add(task.id)
            if dependent not in task_ids:
                raise ValueError(f"Unknown dependency {dependent}")

    queue = deque()
    for taskid, degree in indegree.items():
        if degree == 0:
            queue.append(taskid)

    order = []
    while queue:
        group = []
        for _ in range(len(queue)):
            curtaskid = queue.popleft()
            group.append(curtaskid)
            for nexttask in nexttask_ids[curtaskid]:
                indegree[nexttask] -= 1
                if indegree[nexttask] == 0:
                    queue.append(nexttask)

        order.append(group)

    if sum(len(group) for group in order) != len(plan):
        raise ValueError("Cycle detected in execution plan")

    return order

    

    


    
