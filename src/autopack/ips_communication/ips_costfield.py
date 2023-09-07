import cost_field

def create_ips_field(ips_instance, nodes, geometries, mesh_size = 0.06, clipping_distance = 150):
    # nodes = array with tuples of start, end nodes, and cable diameter (in mm). e.g [(start1, end1, 8),(start2, end2, 10),...]
    # geometries = array with tuples of the geometries, clearance (in mm), how to handle them 0-near, 1-avoid, 2-only prevent collision, and if you can clip. e.g [("Panel",150,0,true), ....]
    nodes_string = ','.join(','.join(str(x) for x in t) for t in nodes)
    geometries_string = ','.join(','.join(str(x) for x in t) for t in geometries)
    command = """
    meshSize = """ + str(mesh_size) + """
    clippingDistance = """ + str(clipping_distance) + """
    nodes_str = \"""" + nodes_string + """\"
    geometries_str = \"""" + geometries_string + """\"
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
    sim:setMinMaxClipClipDist(clippingDistance,clippingDistance*2);
    sim:setMinBoundingBox(false);
    sim:computeGridSize(meshSize);
    sim:buildCostField();

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
    str_cost_field = ips_instance.call(command)
    str_cost_field = str_cost_field.decode('utf-8').strip('"')
    array_cost_field = str_cost_field.split()
    cost_field_size = [int(array_cost_field[0]), int(array_cost_field[1]), int(array_cost_field[2])]
    array_cost_field = array_cost_field[3:]
    cost_field_template = cost_field.CostFieldTemplate(size=cost_field_size)
    cost_field_ips = cost_field.CostField(cost_field_template)
    cost_field_constant = cost_field.CostField(cost_field_template)
    coordinates = [value for i, value in enumerate(array_cost_field) if (i % 4 != 3)]
    costs = [value for i, value in enumerate(array_cost_field) if (i % 4 == 3)]
    cost_field_template.set_coords_from_str_array(coordinates)
    cost_field_ips.set_costs_from_str_array(costs)
    cost_field_constant.costs += 1

    return cost_field_template, cost_field_ips, cost_field_constant



    