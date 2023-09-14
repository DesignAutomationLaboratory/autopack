def optimize_harness(ips_instance, harness_setup, cost_field, bundle_weight = 0.4, harness_number = 0, mesh_size = 0.06, clipping_distance = 150):
    # nodes = array with tuples of start, end nodes, and cable diameter (in mm). e.g [(start1, end1, 8),(start2, end2, 10),...]
    # geometries = array with tuples of the geometries, clearance (in mm), how to handle them 0-near, 1-avoid, 2-only prevent collision, and if you can clip. e.g [("Panel",150,0,true), ....]
    nodes = []
    for cable in harness_setup.cables:
        nodes.append((cable.start_node, cable.end_node, cable.type))
    geometries = []
    for geometry in harness_setup.geometries:
        if geometry.preference == 'Near':
            pref = 0
        elif geometry.preference == 'Avoid':
            pref = 1
        else:
            pref = 2
        geometries.append((geometry.name, geometry.clearance,pref , geometry.clipable))
    cost_field_string = cost_field.get_cost_field_as_str()
    nodes_string = ','.join(','.join(str(x) for x in t) for t in nodes)
    geometries_string = ','.join(','.join(str(x) for x in t) for t in geometries)
    command = """
    meshSize = """ + str(mesh_size) + """
    clippingDistance = """ + str(clipping_distance) + """
    nodes_str = \"""" + nodes_string + """\"
    geometries_str = \"""" + geometries_string + """\"
    costs = \"""" + cost_field_string + """\"
    -- Split function
    function mysplit (inputstr, sep)
        if sep == nil then
                sep = "%s"
        end
        local t={}
        i = 1
        for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
                table.insert(t, str)
        end
        return t
    end
    -- Table length funtion
    function tablelength(T)
        local count = 0
        for _ in pairs(T) do count = count + 1 end
        return count
    end
    -- String to boolean funtion
    function toboolean(str)
        return str == "True"
    end

    -- Create CableComponentTemplate
    local cableSim = CableSimulation();
    local myCableType = cableSim:createComponentTemplate("circular","basic","myTemplate","myTemplate");

    local sim = HarnessRouter();
    
    -- -- GetSurrounding
    local treeObject = Ips.getActiveObjectsRoot(); 

    local surroundings = mysplit(geometries_str,',')
    local nmbOfEntities = tablelength(surroundings)

    for i = 1,nmbOfEntities,4 
    do
        t = treeObject:findFirstMatch(surroundings[i])
        sim:addEnvironmentGeometry(t,tonumber(surroundings[i+1])/1000, tonumber(surroundings[i+2]), toboolean(surroundings[i+3]));
    end

    -- Get start and end points
    local nodes = mysplit(nodes_str,',')
    local treeObject = Ips.getActiveObjectsRoot(); 
    local nmbOfEntities = tablelength(nodes)
    for i = 1,nmbOfEntities,3 
    do        
        local startNode = treeObject:findFirstMatch(nodes[i]);
        local startFrame = startNode:getFirstChild();
        local startVis = startFrame:toCableMountFrameVisualization();
        
        local endNode = treeObject:findFirstMatch(nodes[i+1]);
        local endFrame = endNode:getFirstChild();
        local endVis = endFrame:toCableMountFrameVisualization();
        
        sim:addSegmentTerminalMountFrames(startVis,endVis, myCableType);
    end

    -- Setup Harness
    --sim:setMinMaxClipClipDist(clippingDistance,clippingDistance*2);
    sim:setMinBoundingBox(false);
    sim:computeGridSize(meshSize);
    sim:buildCostField();

    local numbOfCostNodes = sim:getGridSize()
    costs_table = mysplit(costs, ',')
    nmb_of_costs = tablelength(costs_table)

    for n = 1,nmb_of_costs,4
    do
        i = costs_table[n]
        ii = costs_table[n+1]
        iii = costs_table[n+2]
        cost = costs_table[n+3]
        sim:setNodeCost(i, ii, iii, cost)
    end
    --segments = sim:buildDiscreteSegments(0)
    sim:routeHarness();
    if sim:getNumSolutions() == 0 then 
        return
    else
        num = """ + str(bundle_weight) + """*sim:getNumSolutions()
        solution_to_capture = math.floor(num + 0.5)
        segments = sim:buildDiscreteSegments(solution_to_capture)
        static_objects = Ips.getGeometryRoot()
        last_object= static_objects:getLastChild()
        last_object:setLabel("harness" .. """ + str(harness_number) + """);
        nmb_of_segements = segments:size()
        harness = sim:estimateNumClips(solution_to_capture)
        for n = 0,nmb_of_segements-1,1
        do
            segement = sim:getDiscreteSegment(solution_to_capture, n, false)
            elements_in_segment = segement:size()
            print("elements in segment: " .. elements_in_segment)
            radius_of_segment = sim:getSegmentRadius(solution_to_capture, n)
            print("radius of segment: " .. radius_of_segment)
            harness = harness .. "," .. "break"
            for nn = 0,elements_in_segment-1,1
            do
                print(segement[nn])
                harness = harness .. ',' .. segement[nn][0] .. ',' .. segement[nn][1] .. ',' .. segement[nn][2]
            end
        end
        return harness
    end
    """
    segements = ips_instance.call(command)
    segements = segements.decode('utf-8').strip('"').replace("\n", "")[:-1]
    array_segements = segements.split(",")
    nmb_of_clips = array_segements[0]
    array_segements = array_segements[2:]
    segments = identify_segments(array_segements)
    print(segments)
    #print(str_cost_field)

def identify_segments(array_points):
    segments = []
    positions = [idx for idx, s in enumerate(array_points) if s == "break"]
    positions.append(len(array_points))
    start_pos = 0
    for end_pos in positions:
        for pos in range(start_pos,end_pos-3,3):
            if end_pos-start_pos > 4:
                coord1 = [int(array_points[pos]),int(array_points[pos+1]),int(array_points[pos+2])]
                coord2 = [int(array_points[pos+3]),int(array_points[pos+4]),int(array_points[pos+5])]
                segments.append((coord1,coord2))
            else:
                coord1 = [int(array_points[pos]),int(array_points[pos+1]),int(array_points[pos+2])]
                segments.append((coord1))
        start_pos = end_pos+1

   
    print(segments)
    
    return segments


    