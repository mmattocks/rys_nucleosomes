using AWSCore, AWSSDK, Sockets

function spot_wrangle(no_instances::Integer, spot_price::AbstractFloat, sgrp_name::String, sgrp_desc::String, zone::String, ami::String, instance_type::String)
    #ssh permissions
    WANip=chomp(read(pipeline(`dig +short myip.opendns.com @resolver1.opendns.com`), String))
    permissions = [
        [
            "FromPort" => 22,
            "IpProtocol" => "tcp",
            "IpRanges" => [
                [
                    "CidrIp" => WANip*"/32",
                    "Description" => "SSH access"
                ]
            ],
            "ToPort" => 22
        ]
    ]

    @info "Configuring AWS credentials..."
    aws=aws_config()

    @info "Getting security group info..."
    sgrp=Dict()

    try
        sgrp=AWSSDK.EC2.create_security_group(aws,GroupDescription=sgrp_desc,GroupName=sgrp_name)
        inret=AWSSDK.EC2.authorize_security_group_ingress(aws, GroupId=sgrp["groupId"], IpPermissions=permissions)
        outret=AWSSDK.EC2.authorize_security_group_egress(aws, GroupId=sgrp["groupId"], IpPermissions=permissions)
    catch error
        if error.code == "InvalidGroup.Duplicate"
            sgrp=AWSSDK.EC2.describe_security_groups(aws,GroupName=sgrp_name)["securityGroupInfo"]["item"]
        else
            throw(error)
        end
    end

    #Launch specs
    LaunchSpec = [
        "ImageId" => ami,
        "InstanceType" => instance_type,
        "SecurityGroupId" => [
            sgrp["groupId"]
        ],
        "Placement" => [
            "AvailabilityZone" => zone
        ],
        "KeyName" => keys
    ]

    spot_reqs=AWSSDK.EC2.request_spot_instances(aws, LaunchSpecification=LaunchSpec, InstanceCount=no_instances, SpotPrice=spot_price, Type="one-time")

    instance_public_ips=Vector()
    for spot_req in spot_reqs["spotInstanceRequestSet"]["item"]
        spot_req_Id = spot_req["spotInstanceRequestId"]
        active=false
        while !active
            AWSSDK.EC2.describe_spot_instance_requests(aws, SpotInstanceRequestId=spot_req_Id)["spotInstanceRequestSet"]["item"]["state"] == "active" ? active=true : sleep(10)
        end
        instance_Id = AWSSDK.EC2.describe_spot_instance_requests(aws, SpotInstanceRequestId=spot_req_Id)["spotInstanceRequestSet"]["item"]["instanceId"]
        address=AWSSDK.EC2.describe_instances(aws, InstanceId=instance_Id)["reservationSet"]["item"]["instancesSet"]["item"]["ipAddress"]
        push!(instance_public_ips,address)
    end

    return instance_public_ips
end

# function get_cheapest_zone()
#     AWSSDK.EC2.describe_spot_price_history()
# end
