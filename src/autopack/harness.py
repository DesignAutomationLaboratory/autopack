from pydantic import BaseModel


class HarnessSegment(BaseModel):
    cables: list[int]
    points: list[tuple[int, int, int]]

    
class Harness(BaseModel):
    harness_segments: list[HarnessSegment]
    numb_of_clips: int = 0
    

def evaluate_harness(harness, cost_field):
    bundle_cost = 0
    total_cost = 0
    for segment in harness.harness_segments:
        for i in range(len(segment.points)-1):
            start_node = segment.points[i]
            end_node = segment.points[i+1]
            start_coord = cost_field.template.coordinates[start_node[0], start_node[1], start_node[2]]
            end_coord = cost_field.template.coordinates[end_node[0], end_node[1], end_node[2]]
            distance = ((end_coord[0] - start_coord[0])**2 + (end_coord[1] - start_coord[1])**2 + (end_coord[2] - start_coord[2])**2)**0.5
            start_cost = cost_field.costs[start_node[0], start_node[1], start_node[2], 0]
            end_cost = cost_field.costs[end_node[0], end_node[1], end_node[2], 0]
            cost = (start_cost+end_cost)/2*distance
            bundle_cost = bundle_cost + cost
            total_cost = total_cost + cost*len(segment.cables)
    return bundle_cost, total_cost