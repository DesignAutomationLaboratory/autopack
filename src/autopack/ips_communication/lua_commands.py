import numpy as np

from .ips_class import pack


def to_inline_lua(obj):
    """
    Convert a Python value to a representation that can be used as part
    of a Lua statement.
    """
    if isinstance(obj, bool):
        return str(obj).lower()
    # Brute force approach
    return f"autopack.unpack('{pack(obj).decode('utf-8')}')"


def setup_harness_routing(harness):
    command = """
    -- Create CableComponentTemplate
    local cableSim = CableSimulation();
    local sim = HarnessRouter();
    local treeObject = Ips.getActiveObjectsRoot();
    """
    for cable in harness.cables:
        local_command = f"""
        local startNode = treeObject:findFirstMatch('{cable.start_node}');
        local startFrame = startNode:getFirstChild();
        local startVis = startFrame:toCableMountFrameVisualization();
        local endNode = treeObject:findFirstMatch('{cable.end_node}');
        local endFrame = endNode:getFirstChild();
        local endVis = endFrame:toCableMountFrameVisualization();
        local myCableType = cableSim:getComponentTemplate('{cable.cable_type}');
        sim:addSegmentTerminalMountFrames(startVis,endVis, myCableType);
        """
        command = command + local_command

    for geometry in harness.geometries:
        if geometry.preference == "Near":
            pref = 0
        elif geometry.preference == "Avoid":
            pref = 1
        else:
            pref = 2
        local_command = f"""
            local envGeom = treeObject:findFirstMatch('{geometry.name}');
            sim:addEnvironmentGeometry(envGeom, {geometry.clearance}/1000, {pref}, {to_inline_lua(geometry.clipable)});
        """
        command = command + local_command

    command = (
        command
        + f"""
    -- Setup Harness
    sim:setMinMaxClipClipDist({harness.clip_clip_dist[0]},{harness.clip_clip_dist[1]});
    sim:setMinMaxBranchClipDist({harness.branch_clip_dist[0]},{harness.branch_clip_dist[1]})
    sim:setMinBoundingBox(false);
    sim:computeGridSize(0.02);
    --local numbOfCostNodes = sim:getGridSize()
    --print(numbOfCostNodes)
    sim:buildCostField();
    """
    )
    return command


def setup_export_cost_field():
    command = """
    local gridSize = sim:getGridSize()
    local coords = {}
    local costs = {}
    -- Using 1-based indexing to get arrays when packing with msgpack
    for i_x = 1, gridSize[0], 1 do
        coords[i_x] = {}
        costs[i_x] = {}
        for i_y = 1, gridSize[1], 1 do
            coords[i_x][i_y] = {}
            costs[i_x][i_y] = {}
            for i_z = 1, gridSize[2], 1 do
                -- IPS uses 0-based indexing for grid nodes
                local coord = sim:getNodePosition(i_x - 1, i_y - 1, i_z - 1)
                local cost = sim:getNodeCost(i_x - 1, i_y - 1, i_z - 1)
                coords[i_x][i_y][i_z] = {coord.x, coord.y, coord.z}
                costs[i_x][i_y][i_z] = cost
            end
        end
    end
    return autopack.pack({coords=coords, costs=costs})
    """
    return command


def set_node_costs(cost_field):
    commands = []
    for (i_x, i_y, i_z), cost in np.ndenumerate(cost_field.costs):
        cmd = f"sim:setNodeCost({i_x}, {i_y}, {i_z}, {to_inline_lua(cost)})"
        commands.append(cmd)
    new_line = "\n"
    return new_line.join(commands)


def route_harness_one_solution(
    cost_field,
    bundling_factor=0.5,
    harness_id=None,
    build_discrete_solution=False,
    build_presmooth_solution=False,
    build_smooth_solution=False,
):
    return f"""
    {set_node_costs(cost_field)}
    sim:setObjectiveWeights(1, {bundling_factor}, {bundling_factor})
    sim:routeHarness();
    num_solutions = sim:getNumSolutions()
    if num_solutions == 0 then
        return
    else
        solution_to_capture = 0
        nmb_of_segements = sim:getNumBundleSegments(solution_to_capture)
        harness = sim:estimateNumClips(solution_to_capture)
        for n = 0,nmb_of_segements-1,1
        do
            in_seg = sim:getCablesInSegment(solution_to_capture,n)
            -- print(in_seg:size())
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

        if {to_inline_lua(build_discrete_solution)} then
            discreteSegmentsTreeVector = sim:buildDiscreteSegments(solution_to_capture)
            discreteSolution = discreteSegmentsTreeVector[0]:getParent()
            discreteSolution:setLabel("{harness_id} (discrete)")
        end

        if {to_inline_lua(build_presmooth_solution)} then
            presmoothSegmentsTreeVector = sim:buildPresmoothSegments(solution_to_capture)
            presmoothSolution = presmoothSegmentsTreeVector[0]:getParent()
            presmoothSolution:setLabel("{harness_id} (presmooth)")
        end

        if {to_inline_lua(build_smooth_solution)} then
            -- This will smooth ALL available solutions :(
            sim:smoothHarness()
            smoothSegmentsTreeVector = sim:buildSmoothSegments(solution_to_capture, true)
            smoothSolution = smoothSegmentsTreeVector[0]:getParent()
            smoothSolution:setLabel("{harness_id} (smooth)")
        end

        return harness
    end
    """


def route_harness_all_solutions(cost_field, harness_id=None):
    return f"""
    {set_node_costs(cost_field)}
    sim:routeHarness();

    local buildPresmoothSegments = false
    local buildSmoothSegments = false

    local numSolutions = sim:getNumSolutions()
    local solutions = {{}}

    if buildSmoothSegments then
        sim:smoothHarness()
    end

    for sol = 0, numSolutions - 1 do
        local segments = {{}}
        local numSegments = sim:getNumBundleSegments(sol)
        local presmoothSolution
        local smoothSolution
        if buildPresmoothSegments then
            presmoothSolution = sim:buildPresmoothSegments(sol)
        end

        if buildSmoothSegments then
            smoothSolution = sim:buildSmoothSegments(sol, true)
        end

        for seg = 0, numSegments - 1 do
            local segment = {{
                cables = autopack.ipsNVecToTable(sim:getCablesInSegment(sol, seg)),
                discreteNodes = autopack.ipsNVecToTable(sim:getDiscreteSegment(sol, seg, false)),
            }}

            if buildPresmoothSegments then
                segment.presmoothCoords = autopack.ipsNVecToTable(sim:getPresmoothSegment(sol, seg, false))
            end

            if buildSmoothSegments then
                segment.smoothCoords = autopack.ipsNVecToTable(sim:getSmoothSegment(sol, seg, false))
                segment.clipPositions = autopack.ipsNVecToTable(sim:getClipPositions(sol, seg))
            end

            segments[seg + 1] = segment
        end


        -- Gather the solution data
        solutions[sol + 1] = {{
            segments = segments,
            estimatedNumClips = sim:estimateNumClips(sol),
            objectiveWeightBundling = sim:getObjectiveWeightBundling(sol),
            solutionObjectiveBundling = sim:getSolutionObjectiveBundling(sol),
            solutionObjectiveLength = sim:getSolutionObjectiveLength(sol),
        }}
    end

    return autopack.pack(solutions)
    """


def check_coord_distances(measure_dist, harness_setup, coords):
    geos_to_consider = ""
    for geo in harness_setup.geometries:
        if geo.assembly:
            geos_to_consider = geos_to_consider + geo.name + ","
    if len(geos_to_consider) > 0:
        geos_to_consider = geos_to_consider[:-1]
    else:
        return None
    coords_str = ":".join(",".join(map(str, row)) for row in coords)
    command = f"""
    measure_dist = {str(measure_dist)}
    coords = "{coords_str}"
    parts = "{geos_to_consider}"
    function split(source, delimiters)
        local elements = {{}}
        local pattern = '([^'..delimiters..']+)'
        string.gsub(source, pattern, function(value) elements[#elements + 1] =     value;  end);
        return elements
    end
    function tablelength(T)
    local count = 0
    for _ in pairs(T) do count = count + 1 end
    return count
    end

    local treeObject = Ips.getActiveObjectsRoot()
    --sphere = Ips:createSphere(1.0, 1,1)
    prim = PrimitiveShape.createSphere(0.001, 6,6)
    rigid_prim = Ips.createRigidBodyObject(prim)
    geo2 = TreeObjectVector()
    geo2:insert(0, rigid_prim)

    geo = TreeObjectVector()
    parts_table = split(parts,',')
    for i = 1,tablelength(parts_table),1 do
        object = treeObject:findFirstMatch(parts_table[i])
        geo:insert(0, object)
    end



    r = Rot3(Vector3d(0, 0, 0),Vector3d(0, 0, 0),Vector3d(0, 0, 0))
    measure = DistanceMeasure(1,geo, geo2)
    measure_res = ""

    coord_table = split(coords,':')
    for i = 1,tablelength(coord_table),1 do
        local_coords = split(coord_table[i],',')
        local t = Vector3d(tonumber(local_coords[1]), tonumber(local_coords[2]), tonumber(local_coords[3]));
        local trans = Transf3(r, t)
        rigid_prim:setFrameInWorld(trans)
        dist = measure:getDistance()
        if dist<measure_dist-0.01 then
            measure_res = measure_res .. " 1"
        else
            measure_res = measure_res .. " 0"
        end

    end

    Ips.deleteTreeObject(measure)
    Ips.deleteTreeObject(rigid_prim)

    return measure_res
    """

    return command


def get_stl_meshes():
    command = """
    local treeObject = Ips.getActiveObjectsRoot()
    nodes = ""
    local object = treeObject:getFirstChild();
    local numbOfGeoemtries = treeObject:getNumChildren();
    type = object:getType()
    if(type=="RigidBodyObject")
    then
        object1 = object:toPositionedTreeObject()
        object1 = object1:mergeTriangleSubMeshes()
        nodes = "[" .. object:getLabel() .. ",["
        vertices = object1:getVertices()
        numb_verts = vertices:size()
        for n = 0,numb_verts,1
        do
            vert = vertices[n]
            nodes = nodes .. "[" .. tostring(vert[0]) .. tostring(vert[1]) .. tostring(vert[2]) .. "],"
        end
        nodes = nodes .. "],["
        triangles = object1:getTriangles()
        numb_tirangles = vertices:size()
        for n = 0,numb_tirangles,1
        do
            tri = triangles[n]
            nodes = nodes .. "[" .. tostring(tri[0]) .. tostring(tri[1]) .. tostring(tri[2]) .. "],"
        end
        nodes = nodes .. "],"


    end
    for i = 2,numbOfGeoemtries,1
    do
        local objectTemp = object:getNextSibling();
        object = objectTemp;
        type = object:getType()
        if(type=="RigidBodyObject")
        then
            object1 = object:toPositionedTreeObject()
            object1 = object1:mergeTriangleSubMeshes()
            nodes = nodes .. "[" .. object:getLabel() .. ",["
            vertices = object1:getVertices()
            numb_verts = vertices:size()
            for n = 0,numb_verts,1
            do
                vert = vertices[n]
                nodes = nodes .. "[" .. tostring(vert[0]) .. tostring(vert[1]) .. tostring(vert[2]) .. "],"
            end
            nodes = nodes .. "],["
            triangles = object1:getTriangles()
            numb_tirangles = vertices:size()
            for n = 0,numb_tirangles,1
            do
                tri = triangles[n]
                nodes = nodes .. "[" .. tostring(tri[0]) .. tostring(tri[1]) .. tostring(tri[2]) .. "],"
            end
            nodes = nodes .. "],"
        else
            local numbOfchilds = object:getNumChildren();
            local objectobject = object:getFirstChild();
            type = objectobject:getType()
            if(type=="RigidBodyObject")
            then
                object1 = object:toPositionedTreeObject()
                nodes = nodes .. "[" .. object:getLabel() .. ",["
                vertices = object1:getVertices()
                numb_verts = vertices:size()
                for n = 0,numb_verts,1
                do
                    vert = vertices[n]
                    nodes = nodes .. "[" .. tostring(vert[0]) .. tostring(vert[1]) .. tostring(vert[2]) .. "],"
                end
                nodes = nodes .. "],["
                triangles = object1:getTriangles()
                numb_tirangles = vertices:size()
                for n = 0,numb_tirangles,1
                do
                    tri = triangles[n]
                    nodes = nodes .. "[" .. tostring(tri[0]) .. tostring(tri[1]) .. tostring(tri[2]) .. "],"
                end
                nodes = nodes .. "],"
            end
            for ii = 2,numbOfchilds,1
            do
                local objectobjectTemp = objectobject:getNextSibling();
                objectobject = objectobjectTemp;
                type = objectobject:getType()
                name = objectobject:getLabel()
                if(type=="RigidBodyObject")
                then
                    object1 = object:toPositionedTreeObject()
                    nodes = nodes .. "[" .. object:getLabel() .. ",["
                    vertices = object1:getVertices()
                    numb_verts = vertices:size()
                    for n = 0,numb_verts,1
                    do
                        vert = vertices[n]
                        nodes = nodes .. "[" .. tostring(vert[0]) .. tostring(vert[1]) .. tostring(vert[2]) .. "],"
                    end
                    nodes = nodes .. "],["
                    triangles = object1:getTriangles()
                    numb_tirangles = vertices:size()
                    for n = 0,numb_tirangles,1
                    do
                        tri = triangles[n]
                        nodes = nodes .. "[" .. tostring(tri[0]) .. tostring(tri[1]) .. tostring(tri[2]) .. "],"
                    end
                    nodes = nodes .. "],"
                        end
            end
        end
    end
    return nodes
    """
    return command


def ergonomic_evaluation(parts, coords):
    stl_paths = ",".join(parts)
    coords_str = ",".join([" ".join(map(str, sublist)) for sublist in coords])
    command = f"""
    geos = [[{stl_paths}]]
    coordinates = [[{coords_str}]]

    function split(source, delimiters)
            local elements = {{}}
            local pattern = '([^'..delimiters..']+)'
            string.gsub(source, pattern, function(value) elements[#elements + 1] =     value;  end);
            return elements
    end
    function tablelength(T)
        local count = 0
        for _ in pairs(T) do count = count + 1 end
        return count
    end
    function copy_to_static_geometry(part_table)
        numb_of_parts = tablelength(part_table)
        for ii = 1,numb_of_parts,1 do
            part_name = part_table[ii]
            local localtreeobject = Ips.getActiveObjectsRoot();
            local localobject = localtreeobject:findFirstExactMatch(part_name);
            local localrigidObject = localobject:toRigidBodyObject()
            localrigidObject:setLocked(false)
            localnum_of_childs = localrigidObject:getNumChildren()
            localgeometryRoot = Ips.getGeometryRoot()
            for i = 1, localnum_of_childs do
                if i == 1 then
                    localpositionedObject = localrigidObject:getFirstChild()
                    localtoCopy = localpositionedObject:isPositionedTreeObject()

                else
                    localpositionedObject = localpositionedObject:getNextSibling()
                    localtoCopy = localpositionedObject:isPositionedTreeObject()
                end
                if localtoCopy then
                    Ips.copyTreeObject(localpositionedObject, localgeometryRoot)
                end
            end
            --Ips.copyTreeObject
            --positionedObject = object:toPositionedTreeObject()
            localrigidObject:setLocked(true)
        end
    end

    geos_table = split(geos, ",")
    copy_to_static_geometry(geos_table)

    local treeobject = Ips.getActiveObjectsRoot();

    r = Rot3(Vector3d(0, 0, 0),Vector3d(0, 0, 0),Vector3d(0, 0, 0))

    local gp = treeobject:findFirstExactMatch("gp1");
    local gp1=gp:toGripPointVisualization();
    local gp2=gp1:getGripPoint();
    local family = treeobject:findFirstExactMatch("Family 1");
    print(family)
    local f1=family:toManikinFamilyVisualization();
    local f2=f1:getManikinFamily();
    f2:enableCollisionAvoidance();

    measureTree = Ips.getMeasuresRoot()
    measure = measureTree:findFirstExactMatch("measure")
    measure_object = measure:toMeasure()
    print(measure_object)
    local gp_geo = treeobject:findFirstExactMatch("gripGeo");
    gp_geo1 = gp_geo:toPositionedTreeObject()
    --results = ""
    local outputTable = {{}}
    coord_array = split(coordinates, ",")
    numb_of_coords = tablelength(coord_array)
    for i = 1,numb_of_coords,1 do
        coord = split(coord_array[i], " ")
        local trans = Transf3(r, Vector3d(tonumber(coord[1]), tonumber(coord[2]), tonumber(coord[3])))
        gp_geo1:setTControl(trans)
        Ips.moveTreeObject(gp, family);
        dist = measure_object:getValue()
        if dist>0.1 then
            f6_tostring = "99";
            f8_tostring = "99";
        else
            local f4=f2:getErgoStandards();
            local f5=f4[0];
            local f7=f4[1];
            local f6=f2:evaluateStaticErgo(f5, 0);
            local f8=f2:evaluateStaticErgo(f7, 0);
            f6_tostring = tostring(f6);
            f8_tostring = tostring(f8);
        end
        --results = results .. " " .. f6_tostring .. " " .. f8_tostring
        table.insert(outputTable, f6_tostring .. " " .. f8_tostring)
    end
    return table.concat(outputTable, " ")
    --return results
    """
    return command


def add_cost_field_vis(cost_field):
    coords = cost_field.coordinates.reshape(-1, 3)
    norm_cost = cost_field.normalized_costs()
    mask = norm_cost < 9
    max_value = np.amax(norm_cost[mask])
    min_value = np.amin(norm_cost[mask])
    costs = norm_cost.reshape(-1, 1)
    combined_array = np.hstack((coords, costs))
    long_string = " ".join(map(str, combined_array.ravel()))
    command = f"""
    values = [[{long_string}]]
    min_cost = {min_value}
    max_cost = {max_value}

    existing_cost_field = Ips.getGeometryRoot():findFirstExactMatch("CostFieldVis");
    if existing_cost_field ~= nil then
        Ips.deleteTreeObject(existing_cost_field)
    end

    function split(source, delimiters)
        local elements = {{}}
        local pattern = '([^'..delimiters..']+)'
        string.gsub(source, pattern, function(value) elements[#elements + 1] =     value;  end);
        return elements
    end

    function tablelength(T)
        local count = 0
        for _ in pairs(T) do count = count + 1 end
        return count
    end
    -- This is how the heat cost color is computed in harness gui dialog.
    function getRGBheatcost(mincost, maxcost, cost)
        local ratio = (cost - mincost) / (maxcost - mincost)
        if math.abs(maxcost - mincost) < 0.000000001 then -- If max and min cost are equal.
            ratio = 0.5
        end

        local red = math.min(1.0, 2.0 * ratio)
        local green = math.min(1.0, 2.0 * (1.0 - ratio))
        return red, green, 0.0
    end

    value_table = split(values, ' ')
    local builder = GeometryBuilder()

    for i = 1, tablelength(value_table), 4 do
        builder:pushVertex(tonumber(value_table[i]), tonumber(value_table[i+1]), tonumber(value_table[i+2]))
        local cost = tonumber(value_table[i+3])
        if cost > max_cost then
            r, g, b = 0.0, 0.0, 0.0 -- Black color if node is infeasible.
        else
            r, g, b = getRGBheatcost(min_cost, max_cost, cost)
        end
        --
        builder:pushColor(r, g, b)
    end

    builder:buildPoints()
    Ips.getGeometryRoot():getLastChild():setLabel("CostFieldVis")
    """
    return command
