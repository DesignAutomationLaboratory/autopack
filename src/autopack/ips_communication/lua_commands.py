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
