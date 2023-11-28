from typing import Optional

import numpy as np

from ..data_model import CostField
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


def route_harness(
    cost_field: CostField,
    bundling_factor: float,
    case_id: str,
    solutions_to_capture: Optional[list[str]] = None,
    smooth_solutions: bool = False,
    build_discrete_solutions: bool = False,
    build_presmooth_solutions: bool = False,
    build_smooth_solutions: bool = False,
):
    if solutions_to_capture is None:
        solutions_to_capture = []

    return f"""
    local nodeCosts = {to_inline_lua(cost_field.costs)}
    autopack.setHarnessRouterNodeCosts(sim, nodeCosts)
    sim:setObjectiveWeights(1, {bundling_factor}, {bundling_factor})
    sim:routeHarness();

    local buildDiscreteSolutions = {to_inline_lua(build_discrete_solutions)}
    local buildPresmoothSolutions = {to_inline_lua(build_presmooth_solutions)}
    local buildSmoothSolutions = {to_inline_lua(build_smooth_solutions)}
    local smoothSolutions = {to_inline_lua(smooth_solutions)} or buildSmoothSolutions

    local numSolutions = sim:getNumSolutions()
    local solutions = {{}}

    -- To be able to build the smooth segments, this step needs to be run first
    if smoothSolutions then
        sim:smoothHarness()
    end

    local solutionIdxsToCapture = {to_inline_lua(solutions_to_capture)}
    if #solutionIdxsToCapture == 0 then
        -- If no solutions are specified, capture all of them
        solutionIdxsToCapture = autopack.range(0, numSolutions - 1)
    end

    for _, solIdx in pairs(solutionIdxsToCapture) do
        local solutionName = "{case_id}" .. "_" .. solIdx

        local segments = {{}}
        local numSegments = sim:getNumBundleSegments(solIdx)
        for segIdx = 0, numSegments - 1 do
            local segment = {{
                radius = sim:getSegmentRadius(solIdx, segIdx),
                cables = autopack.ipsNVecToTable(sim:getCablesInSegment(solIdx, segIdx)),
                discreteNodes = autopack.ipsNVecToTable(sim:getDiscreteSegment(solIdx, segIdx, false)),
                presmoothCoords = autopack.ipsNVecToTable(sim:getPresmoothSegment(solIdx, segIdx, false)),
                smoothCoords = nil,
                clipPositions = nil,
            }}

            if smoothSolutions then
                -- These are only available if we have run the smoothing step
                segment.smoothCoords = autopack.ipsNVecToTable(sim:getSmoothSegment(solIdx, segIdx, false))
                segment.clipPositions = autopack.ipsNVecToTable(sim:getClipPositions(solIdx, segIdx))
            end

            segments[segIdx + 1] = segment
        end

        if buildDiscreteSolutions then
            builtDiscreteSegmentsTreeVector = sim:buildDiscreteSegments(solIdx)
            builtDiscreteSolution = builtDiscreteSegmentsTreeVector[0]:getParent()
            builtDiscreteSolution:setLabel(solutionName .. " (discrete)")
        end

        if buildPresmoothSolutions then
            builtPresmoothSegmentsTreeVector = sim:buildPresmoothSegments(solIdx)
            builtPresmoothSolution = builtPresmoothSegmentsTreeVector[0]:getParent()
            builtPresmoothSolution:setLabel(solutionName .. " (presmooth)")
        end

        if buildSmoothSolutions then
            builtSmoothSegmentsTreeVector = sim:buildSmoothSegments(solIdx, true)
            builtSmoothSolution = builtSmoothSegmentsTreeVector[0]:getParent()
            builtSmoothSolution:setLabel(solutionName .. " (smooth)")
        end

        -- Gather the solution data
        -- Note that we index by 1 here for packing reasons
        solutions[solIdx + 1] = {{
            name = solutionName,
            segments = segments,
            estimatedNumClips = sim:estimateNumClips(solIdx),
            numBranchPoints = sim:getNumBranchPoints(solIdx),
            objectiveWeightBundling = sim:getObjectiveWeightBundling(solIdx),
            solutionObjectiveBundling = sim:getSolutionObjectiveBundling(solIdx),
            solutionObjectiveLength = sim:getSolutionObjectiveLength(solIdx),
        }}
    end

    return autopack.pack(solutions)
    """


def coord_distances_to_assembly_geo(harness_setup, coords):
    geos_to_consider = [geo.name for geo in harness_setup.geometries if geo.assembly]
    command = f"""
    coords = {to_inline_lua(coords)}
    parts = {to_inline_lua(geos_to_consider)}

    local treeObject = Ips.getActiveObjectsRoot()
    prim = PrimitiveShape.createSphere(0.001, 6, 6)
    rigid_prim = Ips.createRigidBodyObject(prim)
    primTree = TreeObjectVector()
    primTree:insert(0, rigid_prim)

    partsTree = TreeObjectVector()
    for partIdx, part in pairs(parts) do
        partsTree:insert(0, treeObject:findFirstMatch(part))
    end

    r = Rot3(Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(0, 0, 0))
    measure = DistanceMeasure(1, partsTree, primTree)

    distances = {{}}
    for coordIdx, coord in pairs(coords) do
        local trans = Transf3(r, Vector3d(coord[1], coord[2], coord[3]))
        rigid_prim:setFrameInWorld(trans)
        distances[coordIdx] = measure:getDistance()
    end

    Ips.deleteTreeObject(measure)
    Ips.deleteTreeObject(rigid_prim)

    return autopack.pack(distances)
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
    command = f"""
    geo_names = {to_inline_lua(parts)}
    coordinates = {to_inline_lua(coords)}

    function copy_to_static_geometry(part_table)
        for _, part_name in pairs(part_table) do
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
            localrigidObject:setLocked(true)
        end
    end

    copy_to_static_geometry(geo_names)

    local treeobject = Ips.getActiveObjectsRoot();

    local gp = treeobject:findFirstExactMatch("gp1");
    local gp1=gp:toGripPointVisualization();
    local gp2=gp1:getGripPoint();
    local family = treeobject:findFirstExactMatch("Family 1");
    local f1=family:toManikinFamilyVisualization();
    local f2=f1:getManikinFamily();
    f2:enableCollisionAvoidance();
    local representativeManikin = f2:getRepresentative();

    measureTree = Ips.getMeasuresRoot()
    measure = measureTree:findFirstExactMatch("measure")
    measure_object = measure:toMeasure()
    local gp_geo = treeobject:findFirstExactMatch("gripGeo");
    gp_geo1 = gp_geo:toPositionedTreeObject()

    local ergoStandards = autopack.ipsNVecToTable(f2:getErgoStandards())
    local outputTable = {{
        ergoStandards = ergoStandards,
        ergoValues = {{}},
        gripDiffs = {{}},
    }}
    for coordIdx, coord in pairs(coordinates) do
        gp_geo1:transform(coord[1], coord[2], coord[3], 0, 0, 0)
        Ips.moveTreeObject(gp, family);
        f2:posePredict(10)
        -- updateScreen needed for measure to work
        Ips.updateScreen()
        dist = measure_object:getValue()

        local coordErgoValues = {{}}
        for ergoStandardIdx, ergoStandard in pairs(ergoStandards) do
            local ergoValue = f2:evaluateStaticErgo(ergoStandard, representativeManikin)
            coordErgoValues[ergoStandardIdx] = ergoValue
        end
        outputTable.ergoValues[coordIdx] = coordErgoValues
        outputTable.gripDiffs[coordIdx] = dist
    end
    return autopack.pack(outputTable)
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
