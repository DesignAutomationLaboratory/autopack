def setup_harness_routing(harness):
    command = """
    -- Create CableComponentTemplate
    local cableSim = CableSimulation();
    local sim = HarnessRouter();
    local treeObject = Ips.getActiveObjectsRoot(); 
    """
    for cable in harness.cables:
        local_command = """
        local startNode = treeObject:findFirstMatch(\'""" + cable.start_node + """\');
        local startFrame = startNode:getFirstChild();
        local startVis = startFrame:toCableMountFrameVisualization();
        local endNode = treeObject:findFirstMatch(\'""" + cable.end_node + """\');
        local endFrame = endNode:getFirstChild();
        local endVis = endFrame:toCableMountFrameVisualization();
        local myCableType = cableSim:getComponentTemplate(\'""" + cable.cable_type + """\');
        sim:addSegmentTerminalMountFrames(startVis,endVis, myCableType);
        """ 
        command = command + local_command
    
    for geometry in harness.geometries:
        if geometry.preference == 'Near':
            pref = 0
        elif geometry.preference == 'Avoid':
            pref = 1
        else:
            pref = 2
        local_command = """
        t = treeObject:findFirstMatch(\'""" + geometry.name + """\')
        sim:addEnvironmentGeometry(t,""" + str(geometry.clearance) + """/1000, """ + str(pref) + """, """ + bool_to_string_lower(geometry.clipable) + """);
        """
        command = command + local_command

    command = command + """
    -- Setup Harness
    sim:setMinMaxClipClipDist(0.05,0.15);
    sim:setMinBoundingBox(false);
    sim:computeGridSize(0.02);
    --local numbOfCostNodes = sim:getGridSize()
    --print(numbOfCostNodes)
    sim:buildCostField();
    """
    return command

def setup_export_cost_field():
    command = """
    local numbOfCostNodes = sim:getGridSize()
    
    -- Format cost field to string
    output = numbOfCostNodes[0] .. " " .. numbOfCostNodes[1] .. " " .. numbOfCostNodes[2]
    for i = 0,numbOfCostNodes[0]-1,1
    do
        for ii = 0, numbOfCostNodes[1]-1,1
        do
            for iii = 0, numbOfCostNodes[2]-1,1
            do
                local pos = sim:getNodePosition(i,ii,iii)
                output = output .. " " .. pos[0] .. " " .. pos[1] .. " " .. pos[2] .. " " .. sim:getNodeCost(i,ii,iii)
            end
        end
    end
    return output
    """
    return command


def setup_harness_optimization(cost_field, weight=0.5, save_harness=True, harness_id=0):
    commands = []
    
    for i in range(cost_field.template.size[0]):
        for ii in range(cost_field.template.size[1]):
            for iii in range(cost_field.template.size[2]):
                cmd = f"sim:setNodeCost({i}, {ii}, {iii}, {cost_field.costs[i, ii, iii][0]})"
                commands.append(cmd)
    new_line = '\n'
    final_command = f"""
    {new_line.join(commands)}
    sim:routeHarness();
    if sim:getNumSolutions() == 0 then 
        return
    else
        num = {weight}*sim:getNumSolutions()
        solution_to_capture = math.floor(num + 0.5)
        segments = sim:buildDiscreteSegments(solution_to_capture)
        nmb_of_segements = segments:size()
        harness = sim:estimateNumClips(solution_to_capture)
        for n = 0,nmb_of_segements-1,1
        do
            in_seg = sim:getCablesInSegment(solution_to_capture,n)
            print(in_seg:size())
            segement = sim:getDiscreteSegment(solution_to_capture, n, false)
            elements_in_segment = segement:size()
            harness = harness .. "," .. "break" .. "," .. elements_in_segment .. "," .. in_seg:size()
            for nnn = 0,in_seg:size()-1,1
            do
                harness = harness .. "," .. in_seg[nnn]
            end
            for nn = 0,elements_in_segment-1,1
            do
                harness = harness .. ',' .. segement[nn][0] .. ',' .. segement[nn][1] .. ',' .. segement[nn][2]
            end
        end
        static_objects = Ips.getGeometryRoot()
        last_object= static_objects:getLastChild()
        if {bool_to_string_lower(save_harness)} then
            last_object:setLabel("harness{harness_id}");
        else
            Ips.deleteTreeObject(last_object)
        end
        
        return harness
    end
    """
    return final_command

def bool_to_string_lower(bool_val):
    str_val = str(bool_val)
    return str_val[0].lower() + str_val[1:]