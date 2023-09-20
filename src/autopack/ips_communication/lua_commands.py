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

def check_coord_distances(measure_dist, harness_setup, coords):
    geos_to_consider = ""
    for geo in harness_setup.geometries:
        if geo.assembly:
            geos_to_consider = geos_to_consider + geo.name + ","
    if len(geos_to_consider)>0:
        geos_to_consider = geos_to_consider[:-1]
    else:
        return None
    coords_str = ":".join(",".join(map(str, row)) for row in coords)
    command = """
    measure_dist = """ + str(measure_dist) + """
    coords = """ + coords_str + """
    parts = """ + geos_to_consider + """
    function split(source, delimiters)
        local elements = {}
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
    for i = 1,tablelength(parts_table),1 do
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
def bool_to_string_lower(bool_val):
    str_val = str(bool_val)
    return str_val[0].lower() + str_val[1:]