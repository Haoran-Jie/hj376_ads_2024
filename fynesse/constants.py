# Tags to filter
tags = {
    "amenity": ["school", "kindergarten", "college", "university", "library", "childcare", "restaurant", "fast_food", "cafe", "ice_cream", "bar", "pub", "cinema", "theatre", "arts_centre", "nightclub", "bus_station", "bicycle_parking", "bicycle_rental", "community_centre", "events_venue", "place_of_worship", "hospital", "clinic", "pharmacy", "doctors", "dentist", "bank", "post_office", "atm", "marketplace", "charging_station", "toilets", "vending_machine"],
    "landuse": ["farmland", "residential", "grass", "forest", "meadow", "orchard", "farmyard", "industrial", "vineyard", "cemetery", "commercial", "allotments", "retail", "basin", "quarry", "construction", "reservoir", "recreation_ground", "brownfield", "religious", "greenhouse_horticulture", "village_green", "garages", "military", "flowerbed", "railway", "greenfield", "aquaculture", "logging", "plant_nursery", "landfill", "education", "highway", "static_caravan", "salt_pond", "greenery"],
    "building": ["yes", "house", "residential", "detached", "garage", "apartments", "shed", "hut", "industrial", "school", "hospital", "university", "retail", "construction"],
    "highway": ["residential", "service", "track", "footway", "unclassified", "path", "crossing", "tiertiary", "secondary", "street_lamp", "bus_stop", "primary", "turning_circle", "living_street", "cycleway", "stop", "traffic_signals", "trunk", "steps", "motorway"],
    "leisure": ["pitch", "swimming_pool", "garden", "park", "playground", "sports_centre", "stadium"],
    "public_transport": ["station", "stop_position", "platform"],
    "office": ["government", "company", "estate_agent", "educational_institution", "insurance", "it", "telecommunication"],
}

# create a refined tags dictionary that is smaller
refined_tags = {
    "amenity": [
        "school", "university", "restaurant", "pub", "hospital", "bank",
        "pharmacy", "cinema", "place_of_worship", "college", "library",
        "cafeteria", "student_accommodation", "youth_centre", "hostel",
        "bar", "nightclub", "marketplace", "community_centre", "clinic"
    ],
    "landuse": [
        "residential", "industrial", "commercial", "retail", "education",
        "campus", "recreational_ground", "dormitory", "cemetery", "military"
    ],
    "building": [
        "yes", "apartments", "school", "hospital", "retail", "detached",
        "university", "residential", "kindergarten", "campus", "student_hall",
        "library", "hall_of_residence", "dormitory", "terrace", "commercial",
        "hotel", "office"
    ],
    "highway": [
        "residential", "footway", "crossing", "bus_stop", "cycleway",
        "path", "pedestrian", "primary", "secondary", "tertiary"
    ],
    "leisure": [
        "park", "playground", "sports_centre", "stadium", "pitch", "garden",
        "sports_hall", "fitness_centre", "community_centre", "swimming_pool",
        "dog_park", "beach"
    ],
    "public_transport": [
        "station", "stop_position", "platform", "university_stop",
        "bus_station", "tram_stop"
    ],
    "office": [
        "government", "company", "educational_institution", "student_union",
        "career_service", "coworking"
    ],
    "residential": ["university", "student", "dormitory", "mixed"],
}

files_t1 = {
    "oa_coordinates": "./oa_coordinate_geo.csv",
    "ns_sec_oa": "./ns_sec_oa.csv",
    "population_density_oa": "./ts006_oa.csv",
    "oa_feature_counts": "./oa_feature_counts_uk_refined.csv",
    "osm_features": "./osm_features.csv",
    "osm_tags": "./osm_tags.csv"
}

create_oa_coordinates_table = """
CREATE TABLE IF NOT EXISTS oa_coordinates (
    `FID` INT PRIMARY KEY,
    `OA21CD` VARCHAR(20) COLLATE utf8_bin NOT NULL,
    `LSOA21CD` VARCHAR(20) COLLATE utf8_bin NOT NULL,
    `LSOA21NM` VARCHAR(255) COLLATE utf8_bin,
    `LSOA21NMW` VARCHAR(255) COLLATE utf8_bin,
    `BNG_E` INT NOT NULL,
    `BNG_N` INT NOT NULL,
    `LAT` DOUBLE NOT NULL,
    `LONG` DOUBLE NOT NULL,
    `GlobalID` VARCHAR(36) COLLATE utf8_bin NOT NULL,
    `geom` GEOMETRY, -- New column for geometry
    `geometry_wkt` LONGTEXT NOT NULL,      -- Temporary column for WKT geometry
    `area` DOUBLE NOT NULL
) ENGINE=InnoDB;
"""

create_ns_sec_oa_table = """
CREATE TABLE IF NOT EXISTS ns_sec_oa (
    `ns_sec_id` BIGINT(20) AUTO_INCREMENT PRIMARY KEY,
    `date` YEAR NOT NULL,
    `geography` VARCHAR(20) COLLATE utf8_bin NOT NULL,
    `geography_code` VARCHAR(20) COLLATE utf8_bin NOT NULL,
    `total_residents` INT NOT NULL,
    `L1_L3_higher_managerial` INT NOT NULL,
    `L4_L6_lower_managerial` INT NOT NULL,
    `L7_intermediate_occupations` INT NOT NULL,
    `L8_L9_small_employers` INT NOT NULL,
    `L10_L11_lower_supervisory` INT NOT NULL,
    `L12_semi_routine` INT NOT NULL,
    `L13_routine_occupations` INT NOT NULL,
    `L14_never_worked_unemployed` INT NOT NULL,
    `L15_full_time_students` INT NOT NULL,
    UNIQUE (`geography_code`)
);
"""

create_population_density_oa_table = """
CREATE TABLE IF NOT EXISTS population_density_oa (
    `population_density_id` INT AUTO_INCREMENT PRIMARY KEY,
    `date` YEAR NOT NULL,
    `geography` VARCHAR(20) COLLATE utf8_bin NOT NULL,
    `geography_code` VARCHAR(20) COLLATE utf8_bin NOT NULL,
    `population_density_per_sq_km` DOUBLE NOT NULL,
    UNIQUE (`geography_code`)
);
"""

create_oa_feature_counts_table = f"""
CREATE TABLE IF NOT EXISTS oa_feature_counts (
    `FID` INT PRIMARY KEY,
    {", ".join([f"{key}_{value} INT" for key, values in refined_tags.items() for value in values])}
);
"""

create_osm_features = f"""
CREATE TABLE IF NOT EXISTS osm_features (
    id BIGINT PRIMARY KEY,
    type VARCHAR(20) NOT NULL,
    tags TEXT,
    `geometry_wkt` MEDIUMTEXT NOT NULL,
    `geom` GEOMETRY
);
"""

create_osm_tags = f"""
CREATE TABLE IF NOT EXISTS osm_tags (
    osm_feature_id BIGINT NOT NULL,
    `key` VARCHAR(255) NOT NULL,
    value VARCHAR(255) NOT NULL,
    PRIMARY KEY (osm_feature_id, `key`, value),
    FOREIGN KEY (osm_feature_id) REFERENCES osm_features(id)
);
"""

table_definitions_t1 = {
    "oa_coordinates": create_oa_coordinates_table,
    "ns_sec_oa": create_ns_sec_oa_table,
    "population_density_oa": create_population_density_oa_table,
    "oa_feature_counts": create_oa_feature_counts_table,
    "osm_features": create_osm_features,
    "osm_tags": create_osm_tags
}


# Column mappings for CSV loading
column_mappings_t1 = {
    "oa_coordinates": "`FID`, `OA21CD`, `LSOA21CD`, `LSOA21NM`, `LSOA21NMW`, `BNG_E`, `BNG_N`, `LAT`, `LONG`, `GlobalID`, `geometry_wkt`, `area`",
    "ns_sec_oa": "`date`, `geography`, `geography_code`, `total_residents`, `L1_L3_higher_managerial`, `L4_L6_lower_managerial`, `L7_intermediate_occupations`, `L8_L9_small_employers`, `L10_L11_lower_supervisory`, `L12_semi_routine`, `L13_routine_occupations`, `L14_never_worked_unemployed`, `L15_full_time_students`",
    "population_density_oa": "`date`, `geography`, `geography_code`, `population_density_per_sq_km`",
    "oa_feature_counts": f"`FID`, {', '.join([f'{key}_{value}' for key, values in refined_tags.items() for value in values])}",
    "osm_features": "`id`, `type`, `tags`, `geometry_wkt`",
    "osm_tags": "`osm_feature_id`, `key`, `value`"
}

index_definitions_t1 = {
        "oa_coordinates": [["LAT", "LONG"], "OA21CD"],
        "ns_sec_oa": "geography_code",
        "population_density_oa": "geography_code",
        "oa_feature_counts": "FID",
        "osm_tags": [["key", "value"]]
}



















# Define URLs and filenames
file_urls = {
    "constituency_detail_csv": "https://researchbriefings.files.parliament.uk/documents/CBP-10009/HoC-GE2024-results-by-constituency.csv",
    "oa_to_parliamentary_constituency": "https://hub.arcgis.com/api/v3/datasets/5968b5b2c0f14dd29ba277beaae6dec3_0/downloads/data?format=csv&spatialRefId=4326&where=1%3D1",
    "historical_election_results": "https://researchbriefings.files.parliament.uk/documents/CBP-8647/1918-2019election_results.csv",
    "oa_2011_2021_lookup": "https://stg-arcgisazurecdataprod1.az.arcgis.com/exportfiles-1559-19477/OA11_OA21_LAD22_EW_LU_Exact_fit_V2_7175137222568651779.csv?sv=2018-03-28&sr=b&sig=gDG03DFXzI%2Fu%2BwwuLAFtUTx%2Flp6aDfig0LqGK68FzX8%3D&se=2024-11-28T13%3A08%3A27Z&sp=r",
    "oa_2011_pcon_2011": "https://hub.arcgis.com/api/v3/datasets/969687988afe4606815471980042b6fd_0/downloads/data?format=csv&spatialRefId=4326&where=1%3D1",
    "oa_2001_2011_lookup": "https://hub.arcgis.com/api/v3/datasets/40062bd9bcd34cd8b30673847ef52dc3_0/downloads/data?format=csv&spatialRefId=4326&where=1%3D1",
    "oa_2011_pcon_2025": "https://open-geography-portalx-ons.hub.arcgis.com/api/download/v1/items/4491820e4f274755a2553c42fc6a250c/csv?layers=0",
    "pcon25_boundary": "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/Westminster_Parliamentary_Constituencies_July_2024_Boundaries_UK_BFC/FeatureServer/replicafilescache/Westminster_Parliamentary_Constituencies_July_2024_Boundaries_UK_BFC_5018004800687358456.geojson"
}

# Define filenames for each URL
file_names = {
    "constituency_detail_csv": "HoC-GE2024-results-by-constituency.csv",
    "oa_to_parliamentary_constituency": "OA21_PCON25_LU.csv",
    "historical_election_results": "1918-2019election_results.csv",
    "oa_2011_2021_lookup": "OA11_OA21_LAD22_EW_LU_Exact_fit.csv",
    "oa_2011_pcon_2011": "OA11_PCON11_LU.csv",
    "oa_2001_2011_lookup": "OA01_OA11_LU.csv",
    "oa_2011_pcon_2025": "OA11_PCON25_LU.csv",
    "pcon25_boundary": "pcon25_boundary.geojson"
}

election_tags = {
    "amenity": ["polling_station", "place_of_worship", "community_centre", "townhall", "village_hall"],
    "polling_station": ["yes", "ballot_box"],
    "historic": [],
    "leisure": ["park", "playground", "sports_centre", "common"],
}

create_election_results_table = """
CREATE TABLE IF NOT EXISTS election_results (
    ONS_ID VARCHAR(15) COLLATE utf8_bin NOT NULL PRIMARY KEY,
    ONS_region_ID VARCHAR(15) COLLATE utf8_bin NOT NULL,
    Constituency_name VARCHAR(255) COLLATE utf8_bin NOT NULL,
    County_name VARCHAR(255) COLLATE utf8_bin,
    Region_name VARCHAR(255) COLLATE utf8_bin NOT NULL,
    Country_name VARCHAR(255) COLLATE utf8_bin NOT NULL,
    Constituency_type VARCHAR(50) COLLATE utf8_bin NOT NULL,
    Declaration_time VARCHAR(50) COLLATE utf8_bin,
    Member_first_name VARCHAR(255) COLLATE utf8_bin NOT NULL,
    Member_surname VARCHAR(255) COLLATE utf8_bin NOT NULL,
    Member_gender VARCHAR(10) COLLATE utf8_bin NOT NULL,
    Result VARCHAR(50) COLLATE utf8_bin NOT NULL,
    First_party VARCHAR(50) COLLATE utf8_bin NOT NULL,
    Second_party VARCHAR(50) COLLATE utf8_bin NOT NULL,
    Electorate INT NOT NULL,
    Valid_votes INT NOT NULL,
    Invalid_votes INT NOT NULL,
    Majority INT NOT NULL,
    Con INT NOT NULL,
    Lab INT NOT NULL,
    LD INT NOT NULL,
    RUK INT NOT NULL,
    Green INT NOT NULL,
    SNP INT NOT NULL,
    PC INT NOT NULL,
    DUP INT NOT NULL,
    SF INT NOT NULL,
    SDLP INT NOT NULL,
    UUP INT NOT NULL,
    APNI INT NOT NULL,
    All_other_candidates INT NOT NULL,
    Of_which_other_winner INT NOT NULL
);
"""

create_election_results_history_table = """
CREATE TABLE IF NOT EXISTS election_results_history (
    election_id INT AUTO_INCREMENT PRIMARY KEY,
    constituency_id VARCHAR(15),
    seats INT,
    constituency_name VARCHAR(255),
    country_region VARCHAR(255),
    electorate INT,
    con_votes INT,
    con_share FLOAT,
    lib_votes INT,
    lib_share FLOAT,
    lab_votes INT,
    lab_share FLOAT,
    natSW_votes INT,
    natSW_share FLOAT,
    oth_votes INT,
    oth_share FLOAT,
    total_votes INT,
    turnout FLOAT,
    election YEAR,
    boundary_set VARCHAR(50)
);
"""

create_oa_to_constituency_map = """
CREATE TABLE IF NOT EXISTS oa_to_constituency_map (
    ObjectId INT NOT NULL PRIMARY KEY,
    OA21CD VARCHAR(15) COLLATE utf8_bin NOT NULL,
    PCON25CD VARCHAR(15) COLLATE utf8_bin NOT NULL,
    PCON25NM VARCHAR(255) COLLATE utf8_bin,
    PCON25NMW VARCHAR(255) COLLATE utf8_bin,
    LAD21CD VARCHAR(15) COLLATE utf8_bin NOT NULL,
    LAD21NM VARCHAR(255) COLLATE utf8_bin NOT NULL
);
"""

create_oa01_oa11_map = """
CREATE TABLE IF NOT EXISTS oa01_oa11_map (
    OA01CD VARCHAR(15) NOT NULL,
    OA01CDO VARCHAR(15),
    OA11CD VARCHAR(15) NOT NULL,
    CHGIND CHAR(1),
    LAD11CD VARCHAR(15),
    LAD11NM VARCHAR(255),
    LAD11NMW VARCHAR(255),
    ObjectId INT,
    PRIMARY KEY (OA01CD, OA11CD)
);
"""

create_oa11_oa21_map = """
CREATE TABLE IF NOT EXISTS oa11_oa21_map (
    OA11CD VARCHAR(15) NOT NULL,
    OA21CD VARCHAR(15) NOT NULL,
    CHNGIND CHAR(1),
    LAD22CD VARCHAR(15),
    LAD22NM VARCHAR(255),
    LAD22NMW VARCHAR(255),
    ObjectId INT,
    PRIMARY KEY (OA11CD, OA21CD)
);
"""

create_oa11_pcon11_map = """
CREATE TABLE IF NOT EXISTS oa11_pcon11_map (
    OA11CD VARCHAR(15) NOT NULL,
    PCON11CD VARCHAR(15) NOT NULL,
    PCON11NM VARCHAR(255),
    PCON11NMW VARCHAR(255),
    OA11PERCENT FLOAT,
    EER11CD VARCHAR(15),
    EER11NM VARCHAR(255),
    EER11NMW VARCHAR(255),
    ObjectId INT,
    PRIMARY KEY (OA11CD, PCON11CD)
);
"""

create_oa21_pcon25_map = """
CREATE TABLE IF NOT EXISTS oa21_pcon25_map (
    OA21CD VARCHAR(15) NOT NULL,
    PCON25CD VARCHAR(15) NOT NULL,
    PCON25NM VARCHAR(255),
    PCON25NMW VARCHAR(255),
    LAD21CD VARCHAR(15),
    LAD21NM VARCHAR(255),
    ObjectId INT,
    PRIMARY KEY (OA21CD, PCON25CD)
);
"""

create_oa11_pcon25_map = """
CREATE TABLE IF NOT EXISTS oa11_pcon25_map (
    OA11CD VARCHAR(15) NOT NULL,
    PCON25CD VARCHAR(15) NOT NULL,
    PCON25NM VARCHAR(255),
    PCON25NMW VARCHAR(255),
    ObjectId INT,
    PRIMARY KEY (OA11CD, PCON25CD)
);
"""

create_pcon25_boundary = """
CREATE TABLE IF NOT EXISTS pcon25_boundary (
    FID INT NOT NULL,
    PCON25CD VARCHAR(15) NOT NULL PRIMARY KEY,
    PCON25NM VARCHAR(255),
    PCON25NMW VARCHAR(255), 
    BNG_E FLOAT,
    BNG_N FLOAT,
    `LONG` FLOAT,
    `LAT` FLOAT,
    GlobalID VARCHAR(50),
    geom GEOMETRY,
    geometry_wkt MEDIUMTEXT NOT NULL
);
"""

create_pcon_feature_counts = f"""
CREATE TABLE IF NOT EXISTS pcon_feature_counts (
PCON25CD VARCHAR(15) NOT NULL PRIMARY KEY,
{" INT NOT NULL, ".join([f"{key}_{value}" for key, values in election_tags.items() for value in values] + [f"{key}" for key, values in election_tags.items() if not values])} INT NOT NULL
);
"""

create_age_distribution_007a = """
CREATE TABLE IF NOT EXISTS age_distribution_007a (
    age_distribution_id BIGINT(20) AUTO_INCREMENT PRIMARY KEY,
    date YEAR NOT NULL,
    geography VARCHAR(15) COLLATE utf8_bin NOT NULL,
    geography_code VARCHAR(15) COLLATE utf8_bin NOT NULL,
    age_total INT NOT NULL,
    age_4_and_under INT NOT NULL,
    age_5_to_9 INT NOT NULL,
    age_10_to_14 INT NOT NULL,
    age_15_to_19 INT NOT NULL,
    age_20_to_24 INT NOT NULL,
    age_25_to_29 INT NOT NULL,
    age_30_to_34 INT NOT NULL,
    age_35_to_39 INT NOT NULL,
    age_40_to_44 INT NOT NULL,
    age_45_to_49 INT NOT NULL,
    age_50_to_54 INT NOT NULL,
    age_55_to_59 INT NOT NULL,
    age_60_to_64 INT NOT NULL,
    age_65_to_69 INT NOT NULL,
    age_70_to_74 INT NOT NULL,
    age_75_to_79 INT NOT NULL,
    age_80_to_84 INT NOT NULL,
    age_85_and_over INT NOT NULL,
    UNIQUE (`geography_code`)
);
"""

create_age_distribution_2011 = """
CREATE TABLE IF NOT EXISTS age_distribution_2011 (
    age_distribution_id BIGINT(20) AUTO_INCREMENT PRIMARY KEY,
    date YEAR NOT NULL,
    geography VARCHAR(255),
    geography_code VARCHAR(15) NOT NULL,
    rural_urban VARCHAR(50),
    age_all_usual_residents INT,
    age_0_to_4 INT,
    age_5_to_7 INT,
    age_8_to_9 INT,
    age_10_to_14 INT,
    age_15 INT,
    age_16_to_17 INT,
    age_18_to_19 INT,
    age_20_to_24 INT,
    age_25_to_29 INT,
    age_30_to_44 INT,
    age_45_to_59 INT,
    age_60_to_64 INT,
    age_65_to_74 INT,
    age_75_to_84 INT,
    age_85_to_89 INT,
    age_90_and_over INT,
    age_mean FLOAT,
    age_median FLOAT,
    UNIQUE (`geography_code`)
);
""" 

create_age_distribution_2001 = """
CREATE TABLE IF NOT EXISTS age_distribution_2001 (
    age_distribution_id BIGINT(20) AUTO_INCREMENT PRIMARY KEY,
    geography_code VARCHAR(15) NOT NULL,
    all_usual_residents INT NOT NULL,
    age_0_to_4 INT NOT NULL,
    age_5_to_7 INT NOT NULL,
    age_8_to_9 INT NOT NULL,
    age_10_to_14 INT NOT NULL,
    age_15 INT NOT NULL,
    age_16_to_17 INT NOT NULL,
    age_18_to_19 INT NOT NULL,
    age_20_to_24 INT NOT NULL,
    age_25_to_29 INT NOT NULL,
    age_30_to_44 INT NOT NULL,
    age_45_to_59 INT NOT NULL,
    age_60_to_64 INT NOT NULL,
    age_65_to_74 INT NOT NULL,
    age_75_to_84 INT NOT NULL,
    age_85_to_89 INT NOT NULL,
    age_90_and_over INT NOT NULL,
    mean_age FLOAT NOT NULL,
    median_age FLOAT NOT NULL,
    UNIQUE (`geography_code`)
);
"""


create_qualification_distribution_067 = """
CREATE TABLE IF NOT EXISTS qualification_distribution_067 (
    qualification_id BIGINT(20) AUTO_INCREMENT PRIMARY KEY,
    date YEAR NOT NULL,
    geography VARCHAR(15) COLLATE utf8_bin NOT NULL,
    geography_code VARCHAR(15) COLLATE utf8_bin NOT NULL,
    total_usual_residents_16_and_over INT NOT NULL,
    no_qualifications INT NOT NULL,
    level_1_and_entry_level INT NOT NULL,
    level_2_qualifications INT NOT NULL,
    apprenticeship INT NOT NULL,
    level_3_qualifications INT NOT NULL,
    level_4_and_above INT NOT NULL,
    other_qualifications INT NOT NULL,
    UNIQUE (`geography_code`)
);
"""

create_qualification_distribution_2011 = """
CREATE TABLE IF NOT EXISTS qualification_distribution_2011 (
    qualification_distribution_id BIGINT(20) AUTO_INCREMENT PRIMARY KEY,
    date YEAR NOT NULL,
    geography VARCHAR(255),
    geography_code VARCHAR(15) NOT NULL,
    rural_urban VARCHAR(50),
    qualifications_all_categories INT,
    qualifications_no_qualifications INT,
    qualifications_level_1 INT,
    qualifications_level_2 INT,
    qualifications_apprenticeship INT,
    qualifications_level_3 INT,
    qualifications_level_4_and_above INT,
    qualifications_other INT,
    schoolchildren_age_16_to_17 INT,
    schoolchildren_age_18_and_over INT,
    full_time_students_employed INT,
    full_time_students_unemployed INT,
    full_time_students_inactive INT,
    UNIQUE (`geography_code`)
);
"""


create_qualification_distribution_2001 = """
CREATE TABLE IF NOT EXISTS qualification_distribution_2001 (
    qualification_distribution_id BIGINT(20) AUTO_INCREMENT PRIMARY KEY,
    geography_code VARCHAR(15),
    all_people_16_74 INT NOT NULL,
    no_qualifications INT NOT NULL,
    level_1_qualifications INT NOT NULL,
    level_2_qualifications INT NOT NULL,
    level_3_qualifications INT NOT NULL,
    level_4_qualifications_and_above INT NOT NULL,
    other_qualifications INT NOT NULL,
    schoolchildren_16_17 INT NOT NULL,
    schoolchildren_18_74 INT NOT NULL,
    full_time_students_employment INT NOT NULL,
    full_time_students_unemployed INT NOT NULL,
    full_time_students_inactive INT NOT NULL,
    UNIQUE (`geography_code`)
);
"""

create_deprivation_distribution_011 = """
CREATE TABLE IF NOT EXISTS deprivation_distribution_011 (
    deprivation_id BIGINT(20) AUTO_INCREMENT PRIMARY KEY,
    date YEAR NOT NULL,
    geography VARCHAR(15) COLLATE utf8_bin NOT NULL,
    geography_code VARCHAR(15) COLLATE utf8_bin NOT NULL,
    total_households INT NOT NULL,
    not_deprived INT NOT NULL,
    deprived_one_dimension INT NOT NULL,
    deprived_two_dimensions INT NOT NULL,
    deprived_three_dimensions INT NOT NULL,
    deprived_four_dimensions INT NOT NULL,
    UNIQUE (`geography_code`)
);
"""

create_deprivation_distribution_2011 = """
CREATE TABLE IF NOT EXISTS deprivation_distribution_2011 (
    deprivation_distribution_id BIGINT(20) AUTO_INCREMENT PRIMARY KEY,
    date YEAR NOT NULL,
    geography VARCHAR(255),
    geography_code VARCHAR(15) NOT NULL,
    rural_urban VARCHAR(50),
    total_households INT,
    households_not_deprived INT,
    households_deprived_1_dimension INT,
    households_deprived_2_dimensions INT,
    households_deprived_3_dimensions INT,
    households_deprived_4_dimensions INT,
    UNIQUE (`geography_code`)
);
"""

create_deprivation_distribution_2001 = """
CREATE TABLE IF NOT EXISTS deprivation_distribution_2001 (
    deprivation_distribution_id BIGINT(20) AUTO_INCREMENT PRIMARY KEY,
    geography_code VARCHAR(15) NOT NULL,
    all_households INT NOT NULL,
    not_deprived INT NOT NULL,
    deprived_one_dimension INT NOT NULL,
    deprived_two_dimensions INT NOT NULL,
    deprived_three_dimensions INT NOT NULL,
    deprived_four_dimensions INT NOT NULL,
    UNIQUE (`geography_code`)
);
"""

create_economic_activity_distribution_066 = """
CREATE TABLE IF NOT EXISTS economic_activity_distribution_066 (
    economic_activity_id BIGINT(20) AUTO_INCREMENT PRIMARY KEY,
    date YEAR NOT NULL,
    geography VARCHAR(15) COLLATE utf8_bin NOT NULL,
    geography_code VARCHAR(15) COLLATE utf8_bin NOT NULL,
    total_usual_residents_16_and_over INT NOT NULL,
    econ_active_excl_students INT NOT NULL,
    active_employment_excl_students INT NOT NULL,
    active_employee_excl_students INT NOT NULL,
    employee_part_time_excl_students INT NOT NULL,
    employee_full_time_excl_students INT NOT NULL,
    self_employed_with_employees_excl_students INT NOT NULL,
    self_employed_with_employees_part_time INT NOT NULL,
    self_employed_with_employees_full_time INT NOT NULL,
    self_employed_without_employees_excl_students INT NOT NULL,
    self_employed_without_employees_part_time INT NOT NULL,
    self_employed_without_employees_full_time INT NOT NULL,
    active_unemployed_excl_students INT NOT NULL,
    full_time_students INT NOT NULL,
    students_in_employment INT NOT NULL,
    students_employee INT NOT NULL,
    students_employee_part_time INT NOT NULL,
    students_employee_full_time INT NOT NULL,
    students_self_employed_with_employees INT NOT NULL,
    students_self_employed_with_employees_part_time INT NOT NULL,
    students_self_employed_with_employees_full_time INT NOT NULL,
    students_self_employed_without_employees INT NOT NULL,
    students_self_employed_without_employees_part_time INT NOT NULL,
    students_self_employed_without_employees_full_time INT NOT NULL,
    students_unemployed INT NOT NULL,
    econ_inactive INT NOT NULL,
    inactive_retired INT NOT NULL,
    inactive_student INT NOT NULL,
    inactive_home_or_family INT NOT NULL,
    inactive_sick_or_disabled INT NOT NULL,
    inactive_other INT NOT NULL,
    UNIQUE (`geography_code`)
);
"""

create_economic_activity_distribution_2011 = """
CREATE TABLE IF NOT EXISTS economic_activity_distribution_2011 (
    economic_activity_id BIGINT(20) AUTO_INCREMENT PRIMARY KEY,
    geography_code VARCHAR(15) NOT NULL,
    all_usual_residents_16_74 INT NOT NULL,
    economically_active INT NOT NULL,
    economically_active_in_employment INT NOT NULL,
    economically_active_employee_part_time INT NOT NULL,
    economically_active_employee_full_time INT NOT NULL,
    economically_active_self_employed INT NOT NULL,
    economically_active_unemployed INT NOT NULL,
    economically_active_full_time_student INT NOT NULL,
    economically_inactive INT NOT NULL,
    economically_inactive_retired INT NOT NULL,
    economically_inactive_student INT NOT NULL,
    economically_inactive_looking_after_home_or_family INT NOT NULL,
    economically_inactive_long_term_sick_or_disabled INT NOT NULL,
    economically_inactive_other INT NOT NULL,
    unemployed_age_16_24 INT NOT NULL,
    unemployed_age_50_74 INT NOT NULL,
    unemployed_never_worked INT NOT NULL,
    long_term_unemployed INT NOT NULL,
    UNIQUE (`geography_code`)
);
"""

create_economic_activity_distribution_2001 = """
CREATE TABLE IF NOT EXISTS economic_activity_distribution_2001 (
    economic_activity_id BIGINT(20) AUTO_INCREMENT PRIMARY KEY,
    geography_code VARCHAR(15) NOT NULL,
    all_usual_residents_16_74 INT NOT NULL,
    economically_active INT NOT NULL,
    economically_active_in_employment INT NOT NULL,
    economically_active_employee_part_time INT NOT NULL,
    economically_active_employee_full_time INT NOT NULL,
    economically_active_self_employed INT NOT NULL,
    economically_active_unemployed INT NOT NULL,
    economically_active_full_time_student INT NOT NULL,
    economically_inactive INT NOT NULL,
    economically_inactive_retired INT NOT NULL,
    economically_inactive_student INT NOT NULL,
    economically_inactive_looking_after_home_or_family INT NOT NULL,
    economically_inactive_long_term_sick_or_disabled INT NOT NULL,
    economically_inactive_other INT NOT NULL,
    unemployed_age_16_24 INT NOT NULL,
    unemployed_age_50_74 INT NOT NULL,
    unemployed_never_worked INT NOT NULL,
    long_term_unemployed INT NOT NULL,
    UNIQUE (`geography_code`)
);
"""

create_ethnic_group_distribution_021 = """
CREATE TABLE IF NOT EXISTS ethnic_group_distribution_021 (
    ethnic_group_id BIGINT(20) AUTO_INCREMENT PRIMARY KEY,
    date YEAR NOT NULL,
    geography VARCHAR(15) COLLATE utf8_bin NOT NULL,
    geography_code VARCHAR(15) COLLATE utf8_bin NOT NULL,
    total_usual_residents INT NOT NULL,
    asian_total INT NOT NULL,
    asian_bangladeshi INT NOT NULL,
    asian_chinese INT NOT NULL,
    asian_indian INT NOT NULL,
    asian_pakistani INT NOT NULL,
    asian_other INT NOT NULL,
    black_total INT NOT NULL,
    black_african INT NOT NULL,
    black_caribbean INT NOT NULL,
    black_other INT NOT NULL,
    mixed_total INT NOT NULL,
    mixed_white_and_asian INT NOT NULL,
    mixed_white_and_black_african INT NOT NULL,
    mixed_white_and_black_caribbean INT NOT NULL,
    mixed_other INT NOT NULL,
    white_total INT NOT NULL,
    white_british INT NOT NULL,
    white_irish INT NOT NULL,
    white_gypsy_or_irish_traveller INT NOT NULL,
    white_roma INT NOT NULL,
    white_other INT NOT NULL,
    other_total INT NOT NULL,
    other_arab INT NOT NULL,
    other_any_other INT NOT NULL, 
    UNIQUE (`geography_code`)
);
"""

create_ethnic_group_distribution_2011 = """
CREATE TABLE IF NOT EXISTS ethnic_group_distribution_2011 (
    ethnic_group_distribution_id BIGINT(20) AUTO_INCREMENT PRIMARY KEY,
    date YEAR NOT NULL,
    geography VARCHAR(255),
    geography_code VARCHAR(15) NOT NULL,
    rural_urban VARCHAR(50),
    total_usual_residents INT,
    white_total INT,
    white_british INT,
    white_irish INT,
    white_gypsy_traveller INT,
    white_other INT,
    mixed_total INT,
    mixed_white_black_caribbean INT,
    mixed_white_black_african INT,
    mixed_white_asian INT,
    mixed_other INT,
    asian_total INT,
    asian_indian INT,
    asian_pakistani INT,
    asian_bangladeshi INT,
    asian_chinese INT,
    asian_other INT,
    black_total INT,
    black_african INT,
    black_caribbean INT,
    black_other INT,
    other_ethnic_total INT,
    other_ethnic_arab INT,
    other_ethnic_other INT,
    UNIQUE (`geography_code`)
);
"""

create_ethnic_group_distribution_2001 = """
CREATE TABLE IF NOT EXISTS ethnic_group_distribution_2001 (
    `ethnic_group_distribution_id` BIGINT(20) AUTO_INCREMENT PRIMARY KEY,
    `geography_code` VARCHAR(50) NOT NULL,
    all_categories_ethnic_group INT,
    white_total INT,
    white_british INT,
    white_irish INT,
    white_other INT,
    mixed_total INT,
    mixed_white_black_caribbean INT,
    mixed_white_black_african INT,
    mixed_white_asian INT,
    mixed_other INT,
    asian_total INT,
    asian_indian INT,
    asian_pakistani INT,
    asian_bangladeshi INT,
    asian_other INT,
    black_total INT,
    black_black_caribbean INT,
    black_black_african INT,
    black_other INT,
    chinese_other_total INT,
    chinese INT,
    other_ethnic_group INT,
    UNIQUE (`geography_code`)
);
"""

create_household_composition_distribution_003 = """
CREATE TABLE IF NOT EXISTS household_composition_distribution_003 (
    household_composition_id BIGINT(20) AUTO_INCREMENT PRIMARY KEY,
    date YEAR NOT NULL,
    geography VARCHAR(15) COLLATE utf8_bin NOT NULL,
    geography_code VARCHAR(15) COLLATE utf8_bin NOT NULL,
    total_households INT NOT NULL,
    one_person_household INT NOT NULL,
    one_person_aged_66_and_over INT NOT NULL,
    one_person_other INT NOT NULL,
    single_family_household INT NOT NULL,
    family_all_aged_66_and_over INT NOT NULL,
    family_married_couple INT NOT NULL,
    married_no_children INT NOT NULL,
    married_with_dependent_children INT NOT NULL,
    married_with_non_dependent_children INT NOT NULL,
    family_cohabiting_couple INT NOT NULL,
    cohabiting_no_children INT NOT NULL,
    cohabiting_with_dependent_children INT NOT NULL,
    cohabiting_with_non_dependent_children INT NOT NULL,
    family_lone_parent INT NOT NULL,
    lone_parent_with_dependent_children INT NOT NULL,
    lone_parent_with_non_dependent_children INT NOT NULL,
    other_family_household INT NOT NULL,
    other_family_composition INT NOT NULL,
    other_household_types INT NOT NULL,
    other_with_dependent_children INT NOT NULL,
    other_students_and_aged_66_and_over INT NOT NULL,
    UNIQUE (`geography_code`)
);
"""

create_household_composition_distribution_2011 = """
CREATE TABLE IF NOT EXISTS household_composition_distribution_2011 (
    household_composition_id BIGINT(20) AUTO_INCREMENT PRIMARY KEY,
    date YEAR NOT NULL,
    geography VARCHAR(255),
    geography_code VARCHAR(15) NOT NULL,
    rural_urban VARCHAR(50),
    all_households INT,
    one_person_household INT,
    one_person_household_aged_65_and_over INT,
    one_person_household_other INT,
    one_family_household INT,
    one_family_all_aged_65_and_over INT,
    one_family_married_couple INT,
    one_family_married_no_children INT,
    one_family_married_with_dependent_children INT,
    one_family_married_with_non_dependent_children INT,
    one_family_cohabiting_couple INT,
    one_family_cohabiting_no_children INT,
    one_family_cohabiting_with_dependent_children INT,
    one_family_cohabiting_with_non_dependent_children INT,
    one_family_lone_parent INT,
    one_family_lone_parent_with_dependent_children INT,
    one_family_lone_parent_with_non_dependent_children INT,
    other_household_types INT,
    other_households_with_dependent_children INT,
    other_households_all_students INT,
    other_households_all_aged_65_and_over INT,
    other_households_other INT,
    UNIQUE (`geography_code`)
);
"""

create_household_composition_distribution_2001 = """
CREATE TABLE IF NOT EXISTS household_composition_distribution_2001 (
    household_composition_id BIGINT(20) AUTO_INCREMENT PRIMARY KEY,
    geography_code VARCHAR(15) NOT NULL,
    all_households INT NOT NULL,
    one_person_household INT NOT NULL,
    one_person_household_pensioner INT NOT NULL,
    one_person_household_other INT NOT NULL,
    one_family_no_others INT NOT NULL,
    one_family_no_others_all_pensioners INT NOT NULL,
    one_family_no_others_married_no_children INT NOT NULL,
    one_family_no_others_married_with_dependent_children INT NOT NULL,
    one_family_no_others_married_all_children_non_dependent INT NOT NULL,
    one_family_no_others_cohabiting_no_children INT NOT NULL,
    one_family_no_others_cohabiting_with_dependent_children INT NOT NULL,
    one_family_no_others_cohabiting_all_children_non_dependent INT NOT NULL,
    one_family_no_others_lone_parent_with_dependent_children INT NOT NULL,
    one_family_no_others_lone_parent_all_children_non_dependent INT NOT NULL,
    other_households INT NOT NULL,
    other_households_with_dependent_children INT NOT NULL,
    other_households_all_students INT NOT NULL,
    other_households_all_pensioner INT NOT NULL,
    other_households_other INT NOT NULL,
    UNIQUE (`geography_code`)
);
"""

create_table_queries = {
    "age_distribution_007a": create_age_distribution_007a,
    "qualification_distribution_067": create_qualification_distribution_067,
    "deprivation_distribution_011": create_deprivation_distribution_011,
    "economic_activity_distribution_066": create_economic_activity_distribution_066,
    "ethnic_group_distribution_021": create_ethnic_group_distribution_021,
    "household_composition_distribution_003": create_household_composition_distribution_003,
    "election_results": create_election_results_table,
    "oa_to_constituency_map": create_oa_to_constituency_map,
    "age_distribution_2011": create_age_distribution_2011,
    "qualification_distribution_2011": create_qualification_distribution_2011,
    "deprivation_distribution_2011": create_deprivation_distribution_2011,
    "economic_activity_distribution_2011": create_economic_activity_distribution_2011,
    "ethnic_group_distribution_2011": create_ethnic_group_distribution_2011,
    "household_composition_distribution_2011": create_household_composition_distribution_2011,
    "age_distribution_2001": create_age_distribution_2001,
    "qualification_distribution_2001": create_qualification_distribution_2001,
    "deprivation_distribution_2001": create_deprivation_distribution_2001,
    "economic_activity_distribution_2001": create_economic_activity_distribution_2001,
    "ethnic_group_distribution_2001": create_ethnic_group_distribution_2001,
    "household_composition_distribution_2001": create_household_composition_distribution_2001,
    "oa01_oa11_map": create_oa01_oa11_map,
    "oa11_oa21_map": create_oa11_oa21_map,
    "oa11_pcon11_map": create_oa11_pcon11_map,
    "oa21_pcon25_map": create_oa21_pcon25_map,
    "oa11_pcon25_map": create_oa11_pcon25_map,
    "election_results_history": create_election_results_history_table,
    "pcon25_boundary": create_pcon25_boundary,
    "pcon_feature_counts": create_pcon_feature_counts
}

files = {
    'age_distribution_007a': './census2021-ts007a/census2021-ts007a-oa.csv',
    'qualification_distribution_067': './census2021-ts067/census2021-ts067-oa.csv',
    'deprivation_distribution_011': './census2021-ts011/census2021-ts011-oa.csv',
    'economic_activity_distribution_066': './census2021-ts066/census2021-ts066-oa.csv',
    'ethnic_group_distribution_021': './census2021-ts021/census2021-ts021-oa.csv',
    'household_composition_distribution_003': './census2021-ts003/census2021-ts003-oa.csv',
    'election_results': './HoC-GE2024-results-by-constituency.csv',
    'oa_to_constituency_map': './oa_to_parliamentary_constituency.csv',
    'age_distribution_2011': [
        'age_distribution_South_West.csv',
        'age_distribution_South_East.csv',
        'age_distribution_London.csv',
        'age_distribution_West_Midlands.csv',
        'age_distribution_North_West.csv',
        'age_distribution_Yorkshire_Humber.csv',
        'age_distribution_East.csv',
        'age_distribution_Wales.csv',
        'age_distribution_North_East.csv',
        'age_distribution_East_Midlands.csv'
        ],
    'qualification_distribution_2011': [
        'qualification_South_East.csv',
        'qualification_West_Midlands.csv',
        'qualification_South_West.csv',
        'qualification_East.csv',
        'qualification_East_Midlands.csv',
        'qualification_Yorkshire_Humber.csv',
        'qualification_North_East.csv',
        'qualification_Wales.csv',
        'qualification_North_West.csv',
        'qualification_London.csv'
    ],
    'deprivation_distribution_2011': [
        'deprivation_South_West.csv',
        'deprivation_South_East.csv',
        'deprivation_Yorkshire_Humber.csv',
        'deprivation_London.csv',
        'deprivation_East_Midlands.csv',
        'deprivation_North_West.csv',
        'deprivation_East.csv',
        'deprivation_North_East.csv',
        'deprivation_Wales.csv',
        'deprivation_West_Midlands.csv'
    ],
    'economic_activity_distribution_2011': 'economic_activity_distribution_2011.csv',
    'ethnic_group_distribution_2011': [
        'ethnicity_South_East.csv',
        'ethnicity_East.csv',
        'ethnicity_South_West.csv',
        'ethnicity_London.csv',
        'ethnicity_West_Midlands.csv',
        'ethnicity_North_East.csv',
        'ethnicity_Wales.csv',
        'ethnicity_Yorkshire_Humber.csv',
        'ethnicity_East_Midlands.csv',
        'ethnicity_North_West.csv'
    ],
    'household_composition_distribution_2011': [
        'household_composition_North_West.csv',
        'household_composition_North_East.csv',
        'household_composition_Yorkshire_Humber.csv',
        'household_composition_West_Midlands.csv',
        'household_composition_Wales.csv',
        'household_composition_London.csv',
        'household_composition_South_West.csv',
        'household_composition_East.csv',
        'household_composition_South_East.csv',
        'household_composition_East_Midlands.csv'
    ],
    'age_distribution_2001': 'age_distribution_2001.csv',
    'qualification_distribution_2001': 'qualification_distribution_2001.csv',
    'deprivation_distribution_2001': 'deprivation_distribution_2001.csv',
    'economic_activity_distribution_2001': 'economic_activity_distribution_2001.csv',
    'ethnic_group_distribution_2001': 'ethnic_group_2001.csv',
    'household_composition_distribution_2001': 'household_composition_2001.csv',
    'oa01_oa11_map': 'OA01_OA11_LU.csv',
    'oa11_oa21_map': 'OA11_OA21_LAD22_EW_LU_Exact_fit.csv',
    'oa11_pcon11_map': 'OA11_PCON11_LU.csv',
    'oa21_pcon25_map': 'OA21_PCON25_LU.csv',
    'oa11_pcon25_map': 'OA11_PCON25_LU.csv',
    'election_results_history': '1918-2019election_results.csv',
    'pcon25_boundary': 'pcon25_boundary.csv',
    'pcon_feature_counts': 'pcon25_election_features.csv'
}

# Column mappings for CSV loading
column_mappings = {
    "age_distribution_007a": "`date`, `geography`, `geography_code`, `age_total`, `age_4_and_under`, `age_5_to_9`, `age_10_to_14`, `age_15_to_19`, `age_20_to_24`, `age_25_to_29`, `age_30_to_34`, `age_35_to_39`, `age_40_to_44`, `age_45_to_49`, `age_50_to_54`, `age_55_to_59`, `age_60_to_64`, `age_65_to_69`, `age_70_to_74`, `age_75_to_79`, `age_80_to_84`, `age_85_and_over`",

    "qualification_distribution_067": "`date`, `geography`, `geography_code`, `total_usual_residents_16_and_over`, `no_qualifications`, `level_1_and_entry_level`, `level_2_qualifications`, `apprenticeship`, `level_3_qualifications`, `level_4_and_above`, `other_qualifications`",

    "deprivation_distribution_011": "`date`, `geography`, `geography_code`, `total_households`, `not_deprived`, `deprived_one_dimension`, `deprived_two_dimensions`, `deprived_three_dimensions`, `deprived_four_dimensions`",

    "economic_activity_distribution_066": "`date`, `geography`, `geography_code`, `total_usual_residents_16_and_over`, `econ_active_excl_students`, `active_employment_excl_students`, `active_employee_excl_students`, `employee_part_time_excl_students`, `employee_full_time_excl_students`, `self_employed_with_employees_excl_students`, `self_employed_with_employees_part_time`, `self_employed_with_employees_full_time`, `self_employed_without_employees_excl_students`, `self_employed_without_employees_part_time`, `self_employed_without_employees_full_time`, `active_unemployed_excl_students`, `full_time_students`, `students_in_employment`, `students_employee`, `students_employee_part_time`, `students_employee_full_time`, `students_self_employed_with_employees`, `students_self_employed_with_employees_part_time`, `students_self_employed_with_employees_full_time`, `students_self_employed_without_employees`, `students_self_employed_without_employees_part_time`, `students_self_employed_without_employees_full_time`, `students_unemployed`, `econ_inactive`, `inactive_retired`, `inactive_student`, `inactive_home_or_family`, `inactive_sick_or_disabled`, `inactive_other`",

    "ethnic_group_distribution_021": "`date`, `geography`, `geography_code`, `total_usual_residents`, `asian_total`, `asian_bangladeshi`, `asian_chinese`, `asian_indian`, `asian_pakistani`, `asian_other`, `black_total`, `black_african`, `black_caribbean`, `black_other`, `mixed_total`, `mixed_white_and_asian`, `mixed_white_and_black_african`, `mixed_white_and_black_caribbean`, `mixed_other`, `white_total`, `white_british`, `white_irish`, `white_gypsy_or_irish_traveller`, `white_roma`, `white_other`, `other_total`, `other_arab`, `other_any_other`",

    "household_composition_distribution_003": "`date`, `geography`, `geography_code`, `total_households`, `one_person_household`, `one_person_aged_66_and_over`, `one_person_other`, `single_family_household`, `family_all_aged_66_and_over`, `family_married_couple`, `married_no_children`, `married_with_dependent_children`, `married_with_non_dependent_children`, `family_cohabiting_couple`, `cohabiting_no_children`, `cohabiting_with_dependent_children`, `cohabiting_with_non_dependent_children`, `family_lone_parent`, `lone_parent_with_dependent_children`, `lone_parent_with_non_dependent_children`, `other_family_household`, `other_family_composition`, `other_household_types`, `other_with_dependent_children`, `other_students_and_aged_66_and_over`",

    "election_results": "`ONS_ID`, `ONS_region_ID`, `Constituency_name`, `County_name`, `Region_name`, `Country_name`, `Constituency_type`, `Declaration_time`, `Member_first_name`, `Member_surname`, `Member_gender`, `Result`, `First_party`, `Second_party`, `Electorate`, `Valid_votes`, `Invalid_votes`, `Majority`, `Con`, `Lab`, `LD`, `RUK`, `Green`, `SNP`, `PC`, `DUP`, `SF`, `SDLP`, `UUP`, `APNI`, `All_other_candidates`, `Of_which_other_winner`",

    "oa_to_constituency_map": "`OA21CD`, `PCON25CD`, `PCON25NM`, `PCON25NMW`, `LAD21CD`, `LAD21NM`, `ObjectId`",

    "age_distribution_2011": "`date`, `geography`, `geography_code`, `rural_urban`, `age_all_usual_residents`, `age_0_to_4`, `age_5_to_7`, `age_8_to_9`, `age_10_to_14`, `age_15`, `age_16_to_17`, `age_18_to_19`, `age_20_to_24`, `age_25_to_29`, `age_30_to_44`, `age_45_to_59`, `age_60_to_64`, `age_65_to_74`, `age_75_to_84`, `age_85_to_89`, `age_90_and_over`, `age_mean`, `age_median`",

    "age_distribution_2001": "`geography_code`, `all_usual_residents`, `age_0_to_4`, `age_5_to_7`, `age_8_to_9`, `age_10_to_14`, `age_15`, `age_16_to_17`, `age_18_to_19`, `age_20_to_24`, `age_25_to_29`, `age_30_to_44`, `age_45_to_59`, `age_60_to_64`, `age_65_to_74`, `age_75_to_84`, `age_85_to_89`, `age_90_and_over`, `mean_age`, `median_age`",

    "qualification_distribution_2011": "`date`, `geography`, `geography_code`, `rural_urban`, `qualifications_all_categories`, `qualifications_no_qualifications`, `qualifications_level_1`, `qualifications_level_2`, `qualifications_apprenticeship`, `qualifications_level_3`, `qualifications_level_4_and_above`, `qualifications_other`, `schoolchildren_age_16_to_17`, `schoolchildren_age_18_and_over`, `full_time_students_employed`, `full_time_students_unemployed`, `full_time_students_inactive`",

    "qualification_distribution_2001": "`geography_code`, `all_people_16_74`, `no_qualifications`, `level_1_qualifications`, `level_2_qualifications`, `level_3_qualifications`, `level_4_qualifications_and_above`, `other_qualifications`, `schoolchildren_16_17`, `schoolchildren_18_74`, `full_time_students_employment`, `full_time_students_unemployed`, `full_time_students_inactive`",

    "deprivation_distribution_2011": "`date`, `geography`, `geography_code`, `rural_urban`, `total_households`, `households_not_deprived`, `households_deprived_1_dimension`, `households_deprived_2_dimensions`, `households_deprived_3_dimensions`, `households_deprived_4_dimensions`",

    "deprivation_distribution_2001": "`geography_code`, `all_households`, `not_deprived`, `deprived_one_dimension`, `deprived_two_dimensions`, `deprived_three_dimensions`, `deprived_four_dimensions`",

    "economic_activity_distribution_2011": "`geography_code`, `all_usual_residents_16_74`, `economically_active`, `economically_active_in_employment`, `economically_active_employee_part_time`, `economically_active_employee_full_time`, `economically_active_self_employed`, `economically_active_unemployed`, `economically_active_full_time_student`, `economically_inactive`, `economically_inactive_retired`, `economically_inactive_student`, `economically_inactive_looking_after_home_or_family`, `economically_inactive_long_term_sick_or_disabled`, `economically_inactive_other`, `unemployed_age_16_24`, `unemployed_age_50_74`, `unemployed_never_worked`, `long_term_unemployed`",

    "economic_activity_distribution_2001": "`geography_code`, `all_usual_residents_16_74`, `economically_active`, `economically_active_in_employment`, `economically_active_employee_part_time`, `economically_active_employee_full_time`, `economically_active_self_employed`, `economically_active_unemployed`, `economically_active_full_time_student`, `economically_inactive`, `economically_inactive_retired`, `economically_inactive_student`, `economically_inactive_looking_after_home_or_family`, `economically_inactive_long_term_sick_or_disabled`, `economically_inactive_other`, `unemployed_age_16_24`, `unemployed_age_50_74`, `unemployed_never_worked`, `long_term_unemployed`",

    "ethnic_group_distribution_2011": "`date`, `geography`, `geography_code`, `rural_urban`, `total_usual_residents`, `white_total`, `white_british`, `white_irish`, `white_gypsy_traveller`, `white_other`, `mixed_total`, `mixed_white_black_caribbean`, `mixed_white_black_african`, `mixed_white_asian`, `mixed_other`, `asian_total`, `asian_indian`, `asian_pakistani`, `asian_bangladeshi`, `asian_chinese`, `asian_other`, `black_total`, `black_african`, `black_caribbean`, `black_other`, `other_ethnic_total`, `other_ethnic_arab`, `other_ethnic_other`",

    "ethnic_group_distribution_2001": "`geography_code`, `all_categories_ethnic_group`, `white_total`, `white_british`, `white_irish`, `white_other`, `mixed_total`, `mixed_white_black_caribbean`, `mixed_white_black_african`, `mixed_white_asian`, `mixed_other`, `asian_total`, `asian_indian`, `asian_pakistani`, `asian_bangladeshi`, `asian_other`, `black_total`, `black_black_caribbean`, `black_black_african`, `black_other`, `chinese_other_total`, `chinese`, `other_ethnic_group`",

    "household_composition_distribution_2011": "`date`, `geography`, `geography_code`, `rural_urban`, `all_households`, `one_person_household`, `one_person_household_aged_65_and_over`, `one_person_household_other`, `one_family_household`, `one_family_all_aged_65_and_over`, `one_family_married_couple`, `one_family_married_no_children`, `one_family_married_with_dependent_children`, `one_family_married_with_non_dependent_children`, `one_family_cohabiting_couple`, `one_family_cohabiting_no_children`, `one_family_cohabiting_with_dependent_children`, `one_family_cohabiting_with_non_dependent_children`, `one_family_lone_parent`, `one_family_lone_parent_with_dependent_children`, `one_family_lone_parent_with_non_dependent_children`, `other_household_types`, `other_households_with_dependent_children`, `other_households_all_students`, `other_households_all_aged_65_and_over`, `other_households_other`",

    "household_composition_distribution_2001": "`geography_code`, `all_households`, `one_person_household`, `one_person_household_pensioner`, `one_person_household_other`, `one_family_no_others`, `one_family_no_others_all_pensioners`, `one_family_no_others_married_no_children`, `one_family_no_others_married_with_dependent_children`, `one_family_no_others_married_all_children_non_dependent`, `one_family_no_others_cohabiting_no_children`, `one_family_no_others_cohabiting_with_dependent_children`, `one_family_no_others_cohabiting_all_children_non_dependent`, `one_family_no_others_lone_parent_with_dependent_children`, `one_family_no_others_lone_parent_all_children_non_dependent`, `other_households`, `other_households_with_dependent_children`, `other_households_all_students`, `other_households_all_pensioner`, `other_households_other`",

    "oa01_oa11_map": "`OA01CD`, `OA01CDO`, `OA11CD`, `CHGIND`, `LAD11CD`, `LAD11NM`, `LAD11NMW`, `ObjectId`",

    "oa11_oa21_map": "`OA11CD`, `OA21CD`, `CHNGIND`, `LAD22CD`, `LAD22NM`, `LAD22NMW`, `ObjectId`",

    "oa11_pcon11_map": "`OA11CD`, `PCON11CD`, `PCON11NM`, `PCON11NMW`, `OA11PERCENT`, `EER11CD`, `EER11NM`, `EER11NMW`, `ObjectId`",

    "oa21_pcon25_map": "`OA21CD`, `PCON25CD`, `PCON25NM`, `PCON25NMW`, `LAD21CD`, `LAD21NM`, `ObjectId`",

    "oa11_pcon25_map": "`OA11CD`, `PCON25CD`, `PCON25NM`, `PCON25NMW`, `ObjectId`",

    "election_results_history": "`constituency_id`, `seats`, `constituency_name`, `country_region`, `electorate`, `con_votes`, `con_share`, `lib_votes`, `lib_share`, `lab_votes`, `lab_share`, `natSW_votes`, `natSW_share`, `oth_votes`, `oth_share`, `total_votes`, `turnout`, `election`, `boundary_set`",

    "pcon25_boundary": "`FID`, `PCON25CD`, `PCON25NM`, `PCON25NMW`, `BNG_E`, `BNG_N`, `LONG`, `LAT`, `GlobalID`, `geometry_wkt`",

    "pcon_feature_counts": f"`PCON25CD`, {', '.join([f'`{key}_{value}`' for key, values in election_tags.items() for value in values] + [f'`{key}`' for key, values in election_tags.items() if not values])}"
}

indices = {
    "age_distribution_007a": "geography_code",
    "qualification_distribution_067": "geography_code",
    "deprivation_distribution_011": "geography_code",
    "economic_activity_distribution_066": "geography_code",
    "ethnic_group_distribution_021": "geography_code",
    "household_composition_distribution_003": "geography_code",
    "election_results": "ONS_ID",
    "oa_to_constituency_map": ["OA21CD", "PCON25CD"],
    "age_distribution_2011": "geography_code",
    "qualification_distribution_2011": "geography_code",
    "deprivation_distribution_2011": "geography_code",
    "economic_activity_distribution_2011": "geography_code",
    "ethnic_group_distribution_2011": "geography_code",
    "household_composition_distribution_2011": "geography_code",
    "age_distribution_2001": "geography_code",
    "qualification_distribution_2001": "geography_code",
    "deprivation_distribution_2001": "geography_code",
    "economic_activity_distribution_2001": "geography_code",
    "ethnic_group_distribution_2001": "geography_code",
    "household_composition_distribution_2001": "geography_code",
    "oa01_oa11_map": ["OA01CD", "OA11CD"],
    "oa11_oa21_map": ["OA11CD", "OA21CD"],
    "oa11_pcon11_map": ["OA11CD", "PCON11CD"],
    "oa21_pcon25_map": ["OA21CD", "PCON25CD"],
    "oa11_pcon25_map": ["OA11CD", "PCON25CD"],
    "pcon25_boundary": "PCON25CD",
    "pcon_feature_counts": "PCON25CD"
}




COLUMN_PREFIXES = {
    "ethnic_group": ["white_british", "asian_total", "black_total", "mixed_total", "other_total"],
    "household_composition": ["one_person", "lone_parent", "married_couple"],
    "deprivation": ["not_deprived", "deprived_1_dimension", "deprived_2_dimensions", "deprived_3_dimensions", "deprived_4_dimensions"],
    "economic_activity": ["active_employment", "unemployed", "inactive"],
    "qualification": ["no_qualifications", "level_1_qualifications", "level_2_qualifications", "level_3_qualifications", "level_4_and_above", "other_qualifications"]
}

CONFIG = {
    'ethnic_group': {
        'columns': ['white_british', 'asian_total', 'black_total', 'mixed_total', 'other_total'],
        'total_col': 'total_usual_residents'
    },
    'household_composition': {
        'columns': ['one_person', 'lone_parent', 'married_couple'],
        'total_col': 'all_households'
    },
    'deprivation': {
        'columns': ["not_deprived", "deprived_1_dimension", "deprived_2_dimensions", "deprived_3_dimensions", "deprived_4_dimensions"],
        'total_col': 'total_households'
    },
    'qualification': {
        'columns': ["no_qualifications", "level_1_qualifications", "level_2_qualifications", "level_3_qualifications", "level_4_and_above", "other_qualifications"],
        'total_col': 'qualifications_all_categories'
    },
    'economic_activity': {
        'columns': ["active_employment", "unemployed", "inactive"],
        'total_col': 'all_usual_residents_16_74'
    }
}


ethnic_group_direct_query = """
WITH Unchanged_OAs AS (
    SELECT OA01CD, OA11CD, OA21CD 
    FROM oa01_oa21_map
    WHERE CHGIND_01_11 = 'U' AND CHGIND_11_21 = 'U'
)
SELECT 
    uo.OA01CD,
    uo.OA11CD,
    uo.OA21CD,
    -- Proportions for 2001
    e01.white_british / e01.all_categories_ethnic_group AS white_british_2001,
    e01.white_irish / e01.all_categories_ethnic_group AS white_irish_2001,
    e01.white_other / e01.all_categories_ethnic_group AS white_other_2001,
    e01.mixed_total / e01.all_categories_ethnic_group AS mixed_total_2001,
    e01.asian_total / e01.all_categories_ethnic_group AS asian_total_2001,
    e01.black_total / e01.all_categories_ethnic_group AS black_total_2001,
    e01.chinese_other_total / e01.all_categories_ethnic_group AS other_total_2001,
    -- Proportions for 2011
    e11.white_british / e11.total_usual_residents AS white_british_2011,
    e11.white_irish / e11.total_usual_residents AS white_irish_2011,
    e11.white_other / e11.total_usual_residents AS white_other_2011,
    e11.mixed_total / e11.total_usual_residents AS mixed_total_2011,
    e11.asian_total / e11.total_usual_residents AS asian_total_2011,
    e11.black_total / e11.total_usual_residents AS black_total_2011,
    e11.other_ethnic_total / e11.total_usual_residents AS other_total_2011,
    -- Proportions for 2021
    e21.white_british / e21.total_usual_residents AS white_british_2021,
    e21.white_irish / e21.total_usual_residents AS white_irish_2021,
    e21.white_other / e21.total_usual_residents AS white_other_2021,
    e21.mixed_total / e21.total_usual_residents AS mixed_total_2021,
    e21.asian_total / e21.total_usual_residents AS asian_total_2021,
    e21.black_total / e21.total_usual_residents AS black_total_2021,
    e21.other_total / e21.total_usual_residents AS other_total_2021
FROM 
    Unchanged_OAs uo
INNER JOIN 
    ethnic_group_distribution_2001 e01 ON e01.geography_code = uo.OA01CD
INNER JOIN 
    ethnic_group_distribution_2011 e11 ON e11.geography_code = uo.OA11CD
INNER JOIN 
    ethnic_group_distribution_021 e21 ON e21.geography_code = uo.OA21CD;
"""

ethnic_group_approx_query = """
WITH OA_Mapping AS (
    -- Map each 2011 OA to its related 2001 and 2021 OAs
    SELECT 
        oa01.OA11CD,
        oa01.OA01CD,
        oa21.OA21CD
    FROM 
        oa01_oa11_map oa01
    LEFT JOIN 
        oa11_oa21_map oa21 ON oa01.OA11CD = oa21.OA11CD
),
Aggregated_Data AS (
    -- Compute the average proportions for all ethnicities for 2001 and 2021
    SELECT 
        map.OA11CD,
        -- Average proportions for 2001
        AVG(e01.white_british / e01.all_categories_ethnic_group) AS avg_white_british_2001,
        AVG(e01.asian_total / e01.all_categories_ethnic_group) AS avg_asian_total_2001,
        AVG(e01.black_total / e01.all_categories_ethnic_group) AS avg_black_total_2001,
        AVG(e01.mixed_total / e01.all_categories_ethnic_group) AS avg_mixed_total_2001,
        AVG(e01.chinese_other_total / e01.all_categories_ethnic_group) AS avg_chinese_other_total_2001,
        -- Average proportions for 2021
        AVG(e21.white_british / e21.total_usual_residents) AS avg_white_british_2021,
        AVG(e21.asian_total / e21.total_usual_residents) AS avg_asian_total_2021,
        AVG(e21.black_total / e21.total_usual_residents) AS avg_black_total_2021,
        AVG(e21.mixed_total / e21.total_usual_residents) AS avg_mixed_total_2021,
        AVG(e21.other_total / e21.total_usual_residents) AS avg_other_total_2021
    FROM 
        OA_Mapping map
    LEFT JOIN 
        ethnic_group_distribution_2001 e01 ON e01.geography_code = map.OA01CD
    LEFT JOIN 
        ethnic_group_distribution_021 e21 ON e21.geography_code = map.OA21CD
    GROUP BY 
        map.OA11CD
)
SELECT 
    agg.OA11CD,
    -- Proportions for 2001
    agg.avg_white_british_2001 AS white_british_2001,
    agg.avg_asian_total_2001 AS asian_total_2001,
    agg.avg_black_total_2001 AS black_total_2001,
    agg.avg_mixed_total_2001 AS mixed_total_2001,
    agg.avg_chinese_other_total_2001 AS other_total_2001,
    -- Proportions for 2011 (direct match, no aggregation needed)
    e11.white_british / e11.total_usual_residents AS white_british_2011,
    e11.asian_total / e11.total_usual_residents AS asian_total_2011,
    e11.black_total / e11.total_usual_residents AS black_total_2011,
    e11.mixed_total / e11.total_usual_residents AS mixed_total_2011,
    e11.other_ethnic_total / e11.total_usual_residents AS other_total_2011,
    -- Proportions for 2021
    agg.avg_white_british_2021 AS white_british_2021,
    agg.avg_asian_total_2021 AS asian_total_2021,
    agg.avg_black_total_2021 AS black_total_2021,
    agg.avg_mixed_total_2021 AS mixed_total_2021,
    agg.avg_other_total_2021 AS other_total_2021
FROM 
    Aggregated_Data agg
LEFT JOIN 
    ethnic_group_distribution_2011 e11 ON e11.geography_code = agg.OA11CD;
"""

household_composition_direct_query = """
WITH Unchanged_OAs AS (
    SELECT OA01CD, OA11CD, OA21CD 
    FROM oa01_oa21_map
    WHERE CHGIND_01_11 = 'U' AND CHGIND_11_21 = 'U'
)
SELECT 
    uo.OA11CD,
    -- Proportions for 2001
    e01.one_person_household / e01.all_households AS one_person_2001,
    (e01.one_family_no_others_lone_parent_with_dependent_children + e01.one_family_no_others_lone_parent_all_children_non_dependent) / e01.all_households AS lone_parent_2001,
    (e01.one_family_no_others_married_no_children + e01.one_family_no_others_married_with_dependent_children + e01.one_family_no_others_married_all_children_non_dependent) / e01.all_households AS married_couple_2001,
    -- Proportions for 2011
    e11.one_person_household / e11.all_households AS one_person_2011,
    (e11.one_family_lone_parent_with_dependent_children + e11.one_family_lone_parent_with_non_dependent_children) / e11.all_households AS lone_parent_2011,
    (e11.one_family_married_no_children + e11.one_family_married_with_dependent_children + e11.one_family_married_with_non_dependent_children) / e11.all_households AS married_couple_2011,
    -- Proportions for 2021
    e21.one_person_household / e21.total_households AS one_person_2021,
    (e21.lone_parent_with_dependent_children + e21.lone_parent_with_non_dependent_children) / e21.total_households AS lone_parent_2021,
    (e21.married_no_children + e21.married_with_dependent_children + e21.married_with_non_dependent_children) / e21.total_households AS married_couple_2021
FROM 
    Unchanged_OAs uo
INNER JOIN 
    household_composition_distribution_2001 e01 ON e01.geography_code = uo.OA01CD
INNER JOIN 
    household_composition_distribution_2011 e11 ON e11.geography_code = uo.OA11CD
INNER JOIN 
    household_composition_distribution_003 e21 ON e21.geography_code = uo.OA21CD;
"""

household_composition_approx_query = """
WITH OA_Mapping AS (
    -- Map each 2011 OA to its related 2001 and 2021 OAs
    SELECT 
        oa01.OA11CD,
        oa01.OA01CD,
        oa21.OA21CD
    FROM 
        oa01_oa11_map oa01
    LEFT JOIN 
        oa11_oa21_map oa21 ON oa01.OA11CD = oa21.OA11CD
),
Aggregated_Data AS (
    -- Compute the average proportions for all household types for 2001 and 2021
    SELECT 
        map.OA11CD,
        -- Average proportions for 2001
        AVG(e01.one_person_household / e01.all_households) AS avg_one_person_2001,
        AVG((e01.one_family_no_others_lone_parent_with_dependent_children + e01.one_family_no_others_lone_parent_all_children_non_dependent) / e01.all_households) AS avg_lone_parent_2001,
        AVG((e01.one_family_no_others_married_no_children + e01.one_family_no_others_married_with_dependent_children + e01.one_family_no_others_married_all_children_non_dependent) / e01.all_households) AS avg_married_couple_2001,
        -- Average proportions for 2021
        AVG(e21.one_person_household / e21.total_households) AS avg_one_person_2021,
        AVG((e21.lone_parent_with_dependent_children + e21.lone_parent_with_non_dependent_children) / e21.total_households) AS avg_lone_parent_2021,
        AVG((e21.married_no_children + e21.married_with_dependent_children + e21.married_with_non_dependent_children) / e21.total_households) AS avg_married_couple_2021
    FROM 
        OA_Mapping map
    LEFT JOIN 
        household_composition_distribution_2001 e01 ON e01.geography_code = map.OA01CD
    LEFT JOIN 
        household_composition_distribution_003 e21 ON e21.geography_code = map.OA21CD
    GROUP BY 
        map.OA11CD
)
SELECT 
    agg.OA11CD,
    -- Aggregated proportions for 2001
    agg.avg_one_person_2001 AS one_person_2001,
    agg.avg_lone_parent_2001 AS lone_parent_2001,
    agg.avg_married_couple_2001 AS married_couple_2001,
    -- Proportions for 2011 (direct match, no aggregation needed)
    e11.one_person_household / e11.all_households AS one_person_2011,
    (e11.one_family_lone_parent_with_dependent_children + e11.one_family_lone_parent_with_non_dependent_children) / e11.all_households AS lone_parent_2011,
    (e11.one_family_married_no_children + e11.one_family_married_with_dependent_children + e11.one_family_married_with_non_dependent_children) / e11.all_households AS married_couple_2011,
    -- Aggregated proportions for 2021
    agg.avg_one_person_2021 AS one_person_2021,
    agg.avg_lone_parent_2021 AS lone_parent_2021,
    agg.avg_married_couple_2021 AS married_couple_2021
FROM 
    Aggregated_Data agg
LEFT JOIN 
    household_composition_distribution_2011 e11 ON e11.geography_code = agg.OA11CD;
"""

deprivation_direct_query = """
WITH Unchanged_OAs AS (
    SELECT OA01CD, OA11CD, OA21CD 
    FROM oa01_oa21_map
    WHERE CHGIND_01_11 = 'U' AND CHGIND_11_21 = 'U'
)
SELECT 
    uo.OA11CD,
    -- Proportions for 2001
    e01.not_deprived / e01.all_households AS not_deprived_2001,
    e01.deprived_one_dimension / e01.all_households AS deprived_1_dimension_2001,
    e01.deprived_two_dimensions / e01.all_households AS deprived_2_dimensions_2001,
    e01.deprived_three_dimensions / e01.all_households AS deprived_3_dimensions_2001,
    e01.deprived_four_dimensions / e01.all_households AS deprived_4_dimensions_2001,
    -- Proportions for 2011
    e11.households_not_deprived / e11.total_households AS not_deprived_2011,
    e11.households_deprived_1_dimension / e11.total_households AS deprived_1_dimension_2011,
    e11.households_deprived_2_dimensions / e11.total_households AS deprived_2_dimensions_2011,
    e11.households_deprived_3_dimensions / e11.total_households AS deprived_3_dimensions_2011,
    e11.households_deprived_4_dimensions / e11.total_households AS deprived_4_dimensions_2011,
    -- Proportions for 2021
    e21.not_deprived / e21.total_households AS not_deprived_2021,
    e21.deprived_one_dimension / e21.total_households AS deprived_1_dimension_2021,
    e21.deprived_two_dimensions / e21.total_households AS deprived_2_dimensions_2021,
    e21.deprived_three_dimensions / e21.total_households AS deprived_3_dimensions_2021,
    e21.deprived_four_dimensions / e21.total_households AS deprived_4_dimensions_2021
FROM 
    Unchanged_OAs uo
INNER JOIN 
    deprivation_distribution_2001 e01 ON e01.geography_code = uo.OA01CD
INNER JOIN 
    deprivation_distribution_2011 e11 ON e11.geography_code = uo.OA11CD
INNER JOIN 
    deprivation_distribution_011 e21 ON e21.geography_code = uo.OA21CD;
"""

deprivation_approx_query = """
WITH OA_Mapping AS (
    -- Map each 2011 OA to its related 2001 and 2021 OAs
    SELECT 
        oa01.OA11CD,
        oa01.OA01CD,
        oa21.OA21CD
    FROM 
        oa01_oa11_map oa01
    LEFT JOIN 
        oa11_oa21_map oa21 ON oa01.OA11CD = oa21.OA11CD
),
Aggregated_Data AS (
    -- Compute the average proportions for all deprivation categories for 2001 and 2021
    SELECT 
        map.OA11CD,
        -- Average proportions for 2001
        AVG(e01.not_deprived / e01.all_households) AS avg_not_deprived_2001,
        AVG(e01.deprived_one_dimension / e01.all_households) AS avg_deprived_1_dimension_2001,
        AVG(e01.deprived_two_dimensions / e01.all_households) AS avg_deprived_2_dimensions_2001,
        AVG(e01.deprived_three_dimensions / e01.all_households) AS avg_deprived_3_dimensions_2001,
        AVG(e01.deprived_four_dimensions / e01.all_households) AS avg_deprived_4_dimensions_2001,
        -- Average proportions for 2021
        AVG(e21.not_deprived / e21.total_households) AS avg_not_deprived_2021,
        AVG(e21.deprived_one_dimension / e21.total_households) AS avg_deprived_1_dimension_2021,
        AVG(e21.deprived_two_dimensions / e21.total_households) AS avg_deprived_2_dimensions_2021,
        AVG(e21.deprived_three_dimensions / e21.total_households) AS avg_deprived_3_dimensions_2021,
        AVG(e21.deprived_four_dimensions / e21.total_households) AS avg_deprived_4_dimensions_2021
    FROM 
        OA_Mapping map
    LEFT JOIN 
        deprivation_distribution_2001 e01 ON e01.geography_code = map.OA01CD
    LEFT JOIN 
        deprivation_distribution_011 e21 ON e21.geography_code = map.OA21CD
    GROUP BY 
        map.OA11CD
)
SELECT 
    agg.OA11CD,
    -- Aggregated proportions for 2001
    agg.avg_not_deprived_2001 AS not_deprived_2001,
    agg.avg_deprived_1_dimension_2001 AS deprived_1_dimension_2001,
    agg.avg_deprived_2_dimensions_2001 AS deprived_2_dimensions_2001,
    agg.avg_deprived_3_dimensions_2001 AS deprived_3_dimensions_2001,
    agg.avg_deprived_4_dimensions_2001 AS deprived_4_dimensions_2001,
    -- Proportions for 2011 (direct match, no aggregation needed)
    e11.households_not_deprived / e11.total_households AS not_deprived_2011,
    e11.households_deprived_1_dimension / e11.total_households AS deprived_1_dimension_2011,
    e11.households_deprived_2_dimensions / e11.total_households AS deprived_2_dimensions_2011,
    e11.households_deprived_3_dimensions / e11.total_households AS deprived_3_dimensions_2011,
    e11.households_deprived_4_dimensions / e11.total_households AS deprived_4_dimensions_2011,
    -- Aggregated proportions for 2021
    agg.avg_not_deprived_2021 AS not_deprived_2021,
    agg.avg_deprived_1_dimension_2021 AS deprived_1_dimension_2021,
    agg.avg_deprived_2_dimensions_2021 AS deprived_2_dimensions_2021,
    agg.avg_deprived_3_dimensions_2021 AS deprived_3_dimensions_2021,
    agg.avg_deprived_4_dimensions_2021 AS deprived_4_dimensions_2021
FROM 
    Aggregated_Data agg
LEFT JOIN 
    deprivation_distribution_2011 e11 ON e11.geography_code = agg.OA11CD;
"""

economic_activity_direct_query = """
WITH Unchanged_OAs AS (
    SELECT OA01CD, OA11CD, OA21CD 
    FROM oa01_oa21_map
    WHERE CHGIND_01_11 = 'U' AND CHGIND_11_21 = 'U'
)
SELECT 
    uo.OA11CD,
    -- Proportions for 2001
    e01.economically_active_in_employment / e01.all_usual_residents_16_74 AS active_employment_2001,
    e01.economically_active_unemployed / e01.all_usual_residents_16_74 AS unemployed_2001,
    e01.economically_inactive / e01.all_usual_residents_16_74 AS inactive_2001,   
    -- Proportions for 2011
    e11.economically_active_in_employment / e11.all_usual_residents_16_74 AS active_employment_2011,
    e11.economically_active_unemployed / e11.all_usual_residents_16_74 AS unemployed_2011,
    e11.economically_inactive / e11.all_usual_residents_16_74 AS inactive_2011,
     -- Proportions for 2021
    e21.active_employment_excl_students / e21.total_usual_residents_16_and_over AS active_employment_2021,
    e21.active_unemployed_excl_students / e21.total_usual_residents_16_and_over AS unemployed_2021,
    e21.econ_inactive / e21.total_usual_residents_16_and_over AS inactive_2021
FROM 
    Unchanged_OAs uo
INNER JOIN 
    economic_activity_distribution_2001 e01 ON e01.geography_code = uo.OA01CD
INNER JOIN 
    economic_activity_distribution_2011 e11 ON e11.geography_code = uo.OA11CD
INNER JOIN 
    economic_activity_distribution_066 e21 ON e21.geography_code = uo.OA21CD;
"""

economic_activity_approx_query = """
WITH OA_Mapping AS (
    -- Map each 2011 OA to its related 2001 and 2021 OAs
    SELECT 
        oa01.OA11CD,
        oa01.OA01CD,
        oa21.OA21CD
    FROM 
        oa01_oa11_map oa01
    LEFT JOIN 
        oa11_oa21_map oa21 ON oa01.OA11CD = oa21.OA11CD
),
Aggregated_Data AS (
    -- Compute the average proportions for all economic activity categories for 2001 and 2021
    SELECT 
        map.OA11CD,
        -- Average proportions for 2001
        AVG(e01.economically_active_in_employment / e01.all_usual_residents_16_74) AS avg_active_employment_2001,
        AVG(e01.economically_active_unemployed / e01.all_usual_residents_16_74) AS avg_unemployed_2001,
        AVG(e01.economically_inactive / e01.all_usual_residents_16_74) AS avg_inactive_2001,
        -- Average proportions for 2021
        AVG(e21.active_employment_excl_students / e21.total_usual_residents_16_and_over) AS avg_active_employment_2021,
        AVG(e21.active_unemployed_excl_students / e21.total_usual_residents_16_and_over) AS avg_unemployed_2021,
        AVG(e21.econ_inactive / e21.total_usual_residents_16_and_over) AS avg_inactive_2021
    FROM 
        OA_Mapping map
    LEFT JOIN 
        economic_activity_distribution_2001 e01 ON e01.geography_code = map.OA01CD
    LEFT JOIN 
        economic_activity_distribution_066 e21 ON e21.geography_code = map.OA21CD
    GROUP BY 
        map.OA11CD
)
SELECT 
    agg.OA11CD,
    -- Aggregated proportions for 2001
    agg.avg_active_employment_2001 AS active_employment_2001,
    agg.avg_unemployed_2001 AS unemployed_2001,
    agg.avg_inactive_2001 AS inactive_2001,
    -- Proportions for 2011 (direct match, no aggregation needed)
    e11.economically_active_in_employment / e11.all_usual_residents_16_74 AS active_employment_2011,
    e11.economically_active_unemployed / e11.all_usual_residents_16_74 AS unemployed_2011,
    e11.economically_inactive / e11.all_usual_residents_16_74 AS inactive_2011,
    -- Aggregated proportions for 2021
    agg.avg_active_employment_2021 AS active_employment_2021,
    agg.avg_unemployed_2021 AS unemployed_2021,
    agg.avg_inactive_2021 AS inactive_2021
FROM 
    Aggregated_Data agg
LEFT JOIN 
    economic_activity_distribution_2011 e11 ON e11.geography_code = agg.OA11CD;
"""

qualification_direct_query = """
WITH Unchanged_OAs AS (
    SELECT OA01CD, OA11CD, OA21CD 
    FROM oa01_oa21_map
    WHERE CHGIND_01_11 = 'U' AND CHGIND_11_21 = 'U'
)
SELECT 
    uo.OA11CD,
    -- Proportions for 2001
    e01.no_qualifications / e01.all_people_16_74 AS no_qualifications_2001,
    e01.level_1_qualifications / e01.all_people_16_74 AS level_1_qualifications_2001,
    e01.level_2_qualifications / e01.all_people_16_74 AS level_2_qualifications_2001,
    e01.level_3_qualifications / e01.all_people_16_74 AS level_3_qualifications_2001,
    e01.level_4_qualifications_and_above / e01.all_people_16_74 AS level_4_and_above_2001,
    e01.other_qualifications / e01.all_people_16_74 AS other_qualifications_2001,
    -- Proportions for 2011
    e11.qualifications_no_qualifications / e11.qualifications_all_categories AS no_qualifications_2011,
    e11.qualifications_level_1 / e11.qualifications_all_categories AS level_1_qualifications_2011,
    e11.qualifications_level_2 / e11.qualifications_all_categories AS level_2_qualifications_2011,
    e11.qualifications_level_3 / e11.qualifications_all_categories AS level_3_qualifications_2011,
    e11.qualifications_level_4_and_above / e11.qualifications_all_categories AS level_4_and_above_2011,
    e11.qualifications_other / e11.qualifications_all_categories AS other_qualifications_2011,
    -- Proportions for 2021
    e21.no_qualifications / e21.total_usual_residents_16_and_over AS no_qualifications_2021,
    e21.level_1_and_entry_level / e21.total_usual_residents_16_and_over AS level_1_qualifications_2021,
    e21.level_2_qualifications / e21.total_usual_residents_16_and_over AS level_2_qualifications_2021,
    e21.level_3_qualifications / e21.total_usual_residents_16_and_over AS level_3_qualifications_2021,
    e21.level_4_and_above / e21.total_usual_residents_16_and_over AS level_4_and_above_2021,
    e21.other_qualifications / e21.total_usual_residents_16_and_over AS other_qualifications_2021
FROM 
    Unchanged_OAs uo
INNER JOIN 
    qualification_distribution_2001 e01 ON e01.geography_code = uo.OA01CD
INNER JOIN 
    qualification_distribution_2011 e11 ON e11.geography_code = uo.OA11CD
INNER JOIN 
    qualification_distribution_067 e21 ON e21.geography_code = uo.OA21CD;
"""

qualification_approx_query = """
WITH OA_Mapping AS (
    -- Map each 2011 OA to its related 2001 and 2021 OAs
    SELECT 
        oa01.OA11CD,
        oa01.OA01CD,
        oa21.OA21CD
    FROM 
        oa01_oa11_map oa01
    LEFT JOIN 
        oa11_oa21_map oa21 ON oa01.OA11CD = oa21.OA11CD
),
Aggregated_Data AS (
    -- Compute the average proportions for all qualification categories for 2001 and 2021
    SELECT 
        map.OA11CD,
        -- Average proportions for 2001
        AVG(e01.no_qualifications / e01.all_people_16_74) AS avg_no_qualifications_2001,
        AVG(e01.level_1_qualifications / e01.all_people_16_74) AS avg_level_1_qualifications_2001,
        AVG(e01.level_2_qualifications / e01.all_people_16_74) AS avg_level_2_qualifications_2001,
        AVG(e01.level_3_qualifications / e01.all_people_16_74) AS avg_level_3_qualifications_2001,
        AVG(e01.level_4_qualifications_and_above / e01.all_people_16_74) AS avg_level_4_and_above_2001,
        AVG(e01.other_qualifications / e01.all_people_16_74) AS avg_other_qualifications_2001,

        -- Average proportions for 2021
        AVG(e21.no_qualifications / e21.total_usual_residents_16_and_over) AS avg_no_qualifications_2021,
        AVG(e21.level_1_and_entry_level / e21.total_usual_residents_16_and_over) AS avg_level_1_qualifications_2021,
        AVG(e21.level_2_qualifications / e21.total_usual_residents_16_and_over) AS avg_level_2_qualifications_2021,
        AVG(e21.level_3_qualifications / e21.total_usual_residents_16_and_over) AS avg_level_3_qualifications_2021,
        AVG(e21.level_4_and_above / e21.total_usual_residents_16_and_over) AS avg_level_4_and_above_2021,
        AVG(e21.other_qualifications / e21.total_usual_residents_16_and_over) AS avg_other_qualifications_2021
    FROM 
        OA_Mapping map
    LEFT JOIN 
        qualification_distribution_2001 e01 ON e01.geography_code = map.OA01CD
    LEFT JOIN 
        qualification_distribution_067 e21 ON e21.geography_code = map.OA21CD
    GROUP BY 
        map.OA11CD
)
SELECT 
    agg.OA11CD,
    -- Aggregated proportions for 2001
    agg.avg_no_qualifications_2001 AS no_qualifications_2001,
    agg.avg_level_1_qualifications_2001 AS level_1_qualifications_2001,
    agg.avg_level_2_qualifications_2001 AS level_2_qualifications_2001,
    agg.avg_level_3_qualifications_2001 AS level_3_qualifications_2001,
    agg.avg_level_4_and_above_2001 AS level_4_and_above_2001,
    agg.avg_other_qualifications_2001 AS other_qualifications_2001,
    -- Proportions for 2011 (direct match, no aggregation needed)
    e11.qualifications_no_qualifications / e11.qualifications_all_categories AS no_qualifications_2011,
    e11.qualifications_level_1 / e11.qualifications_all_categories AS level_1_qualifications_2011,
    e11.qualifications_level_2 / e11.qualifications_all_categories AS level_2_qualifications_2011,
    e11.qualifications_level_3 / e11.qualifications_all_categories AS level_3_qualifications_2011,
    e11.qualifications_level_4_and_above / e11.qualifications_all_categories AS level_4_and_above_2011,
    e11.qualifications_other / e11.qualifications_all_categories AS other_qualifications_2011,
    -- Aggregated proportions for 2021
    agg.avg_no_qualifications_2021 AS no_qualifications_2021,
    agg.avg_level_1_qualifications_2021 AS level_1_qualifications_2021,
    agg.avg_level_2_qualifications_2021 AS level_2_qualifications_2021,
    agg.avg_level_3_qualifications_2021 AS level_3_qualifications_2021,
    agg.avg_level_4_and_above_2021 AS level_4_and_above_2021,
    agg.avg_other_qualifications_2021 AS other_qualifications_2021
FROM 
    Aggregated_Data agg
LEFT JOIN 
    qualification_distribution_2011 e11 ON e11.geography_code = agg.OA11CD;
"""

DIRECT_APPROX_QUERIES = {
    "ethnic_group": (ethnic_group_direct_query, ethnic_group_approx_query),
    "household_composition": (household_composition_direct_query, household_composition_approx_query),
    "deprivation": (deprivation_direct_query, deprivation_approx_query),
    "economic_activity": (economic_activity_direct_query, economic_activity_approx_query),
    "qualification": (qualification_direct_query, qualification_approx_query)
}


CTE_NAMES_LIST = [
    "age_distribution_per_constituency",
    "qualification_distribution_per_constituency",
    "deprivation_distribution_per_constituency",
    "economic_activity_distribution_per_constituency",
    "ethnic_group_distribution_per_constituency",
    "household_composition_distribution_per_constituency"
]




age_distribution_per_constituency_query = """
SELECT 
    o.PCON25CD AS constituency_id,
    SUM(a.age_total) AS age_total,
    SUM(a.age_4_and_under) / SUM(a.age_total) AS prop_4_and_under,
    SUM(a.age_5_to_9) / SUM(a.age_total) AS prop_5_to_9,
    SUM(a.age_10_to_14) / SUM(a.age_total) AS prop_10_to_14,
    SUM(a.age_15_to_19) / SUM(a.age_total) AS prop_15_to_19,
    SUM(a.age_20_to_24) / SUM(a.age_total) AS prop_20_to_24,
    SUM(a.age_25_to_29) / SUM(a.age_total) AS prop_25_to_29,
    SUM(a.age_30_to_34) / SUM(a.age_total) AS prop_30_to_34,
    SUM(a.age_35_to_39) / SUM(a.age_total) AS prop_35_to_39,
    SUM(a.age_40_to_44) / SUM(a.age_total) AS prop_40_to_44,
    SUM(a.age_45_to_49) / SUM(a.age_total) AS prop_45_to_49,
    SUM(a.age_50_to_54) / SUM(a.age_total) AS prop_50_to_54,
    SUM(a.age_55_to_59) / SUM(a.age_total) AS prop_55_to_59,
    SUM(a.age_60_to_64) / SUM(a.age_total) AS prop_60_to_64,
    SUM(a.age_65_to_69) / SUM(a.age_total) AS prop_65_to_69,
    SUM(a.age_70_to_74) / SUM(a.age_total) AS prop_70_to_74,
    SUM(a.age_75_to_79) / SUM(a.age_total) AS prop_75_to_79,
    SUM(a.age_80_to_84) / SUM(a.age_total) AS prop_80_to_84,
    SUM(a.age_85_and_over) / SUM(a.age_total) AS prop_85_and_over
FROM 
    oa_to_constituency_map o
INNER JOIN 
    age_distribution_007a a
ON 
    o.OA21CD = a.geography_code
GROUP BY 
    o.PCON25CD
"""

qualification_distribution_per_constituency_query = """
SELECT 
    o.PCON25CD AS constituency_id,
    SUM(q.total_usual_residents_16_and_over) AS total_16_and_over,
    SUM(q.no_qualifications) / SUM(q.total_usual_residents_16_and_over) AS prop_no_qualifications,
    SUM(q.level_1_and_entry_level) / SUM(q.total_usual_residents_16_and_over) AS prop_level_1_and_entry_level,
    SUM(q.level_2_qualifications) / SUM(q.total_usual_residents_16_and_over) AS prop_level_2_qualifications,
    SUM(q.apprenticeship) / SUM(q.total_usual_residents_16_and_over) AS prop_apprenticeship,
    SUM(q.level_3_qualifications) / SUM(q.total_usual_residents_16_and_over) AS prop_level_3_qualifications,
    SUM(q.level_4_and_above) / SUM(q.total_usual_residents_16_and_over) AS prop_level_4_and_above,
    SUM(q.other_qualifications) / SUM(q.total_usual_residents_16_and_over) AS prop_other_qualifications
FROM 
    oa_to_constituency_map o
INNER JOIN 
    qualification_distribution_067 q
ON 
    o.OA21CD = q.geography_code
GROUP BY 
    o.PCON25CD
"""

deprivation_distribution_per_constituency_query = """
SELECT 
    o.PCON25CD AS constituency_id,
    SUM(d.total_households) AS total_households,
    SUM(d.not_deprived) / SUM(d.total_households) AS prop_not_deprived,
    SUM(d.deprived_one_dimension) / SUM(d.total_households) AS prop_deprived_one_dimension,
    SUM(d.deprived_two_dimensions) / SUM(d.total_households) AS prop_deprived_two_dimensions,
    SUM(d.deprived_three_dimensions) / SUM(d.total_households) AS prop_deprived_three_dimensions,
    SUM(d.deprived_four_dimensions) / SUM(d.total_households) AS prop_deprived_four_dimensions
FROM 
    oa_to_constituency_map o
INNER JOIN 
    deprivation_distribution_011 d
ON 
    o.OA21CD = d.geography_code
GROUP BY 
    o.PCON25CD
"""

economic_activity_distribution_per_constituency_query = """
SELECT 
    o.PCON25CD AS constituency_id,
    SUM(e.total_usual_residents_16_and_over) AS total_population_16_and_over,

    -- Active employment proportions
    (SUM(e.active_employment_excl_students) / SUM(e.total_usual_residents_16_and_over)) AS prop_active_employment,
    (SUM(e.employee_part_time_excl_students) + SUM(e.employee_full_time_excl_students)) / SUM(e.total_usual_residents_16_and_over) AS prop_active_employees,
    (SUM(e.self_employed_with_employees_excl_students) + SUM(e.self_employed_without_employees_excl_students)) / SUM(e.total_usual_residents_16_and_over) AS prop_self_employed,

    -- Unemployment
    SUM(e.active_unemployed_excl_students) / SUM(e.total_usual_residents_16_and_over) AS prop_unemployed,

    -- Students in economic activity
    (SUM(e.students_in_employment) + SUM(e.students_employee) + SUM(e.students_self_employed_without_employees)) / SUM(e.total_usual_residents_16_and_over) AS prop_students_in_employment,

    -- Full-time students not in economic activity
    SUM(e.full_time_students) / SUM(e.total_usual_residents_16_and_over) AS prop_full_time_students,

    -- Economically inactive
    SUM(e.econ_inactive) / SUM(e.total_usual_residents_16_and_over) AS prop_inactive,
    SUM(e.inactive_retired) / SUM(e.total_usual_residents_16_and_over) AS prop_inactive_retired,
    SUM(e.inactive_student) / SUM(e.total_usual_residents_16_and_over) AS prop_inactive_student,
    SUM(e.inactive_home_or_family) / SUM(e.total_usual_residents_16_and_over) AS prop_inactive_home_or_family,
    SUM(e.inactive_sick_or_disabled) / SUM(e.total_usual_residents_16_and_over) AS prop_inactive_sick_or_disabled

FROM 
    oa_to_constituency_map o
INNER JOIN 
    economic_activity_distribution_066 e
ON 
    o.OA21CD = e.geography_code
GROUP BY 
    o.PCON25CD
"""

ethnic_group_distribution_per_constituency_query = """
SELECT 
    o.PCON25CD AS constituency_id,
    SUM(e.total_usual_residents) AS total_population,

    -- Major ethnic group proportions
    SUM(e.asian_total) / SUM(e.total_usual_residents) AS prop_asian,
    SUM(e.black_total) / SUM(e.total_usual_residents) AS prop_black,
    SUM(e.mixed_total) / SUM(e.total_usual_residents) AS prop_mixed,
    SUM(e.white_total) / SUM(e.total_usual_residents) AS prop_white,
    SUM(e.other_total) / SUM(e.total_usual_residents) AS prop_other,

    -- Selected Asian subgroups
    SUM(e.asian_indian) / SUM(e.total_usual_residents) AS prop_asian_indian,
    SUM(e.asian_pakistani) / SUM(e.total_usual_residents) AS prop_asian_pakistani,

    -- Selected Black subgroups
    SUM(e.black_african) / SUM(e.total_usual_residents) AS prop_black_african,
    SUM(e.black_caribbean) / SUM(e.total_usual_residents) AS prop_black_caribbean,

    -- Selected Mixed subgroups
    SUM(e.mixed_white_and_asian) / SUM(e.total_usual_residents) AS prop_mixed_white_and_asian,
    SUM(e.mixed_white_and_black_african) / SUM(e.total_usual_residents) AS prop_mixed_white_and_black_african,

    -- Selected White subgroups
    SUM(e.white_british) / SUM(e.total_usual_residents) AS prop_white_british,
    SUM(e.white_other) / SUM(e.total_usual_residents) AS prop_white_other

FROM 
    oa_to_constituency_map o
INNER JOIN 
    ethnic_group_distribution_021 e
ON 
    o.OA21CD = e.geography_code
GROUP BY 
    o.PCON25CD
"""

household_composition_distribution_per_constituency_query = """
SELECT 
    o.PCON25CD AS constituency_id,
    SUM(h.total_households) AS total_households,

    -- One-person households
    SUM(h.one_person_household) / SUM(h.total_households) AS prop_one_person,
    SUM(h.one_person_aged_66_and_over) / SUM(h.total_households) AS prop_one_person_aged_66_and_over,
    SUM(h.one_person_other) / SUM(h.total_households) AS prop_one_person_other,

    -- Single-family households
    SUM(h.single_family_household) / SUM(h.total_households) AS prop_single_family,
    SUM(h.family_all_aged_66_and_over) / SUM(h.total_households) AS prop_family_all_aged_66_and_over,
    SUM(h.family_married_couple) / SUM(h.total_households) AS prop_family_married_couple,
    SUM(h.family_cohabiting_couple) / SUM(h.total_households) AS prop_family_cohabiting_couple,
    SUM(h.family_lone_parent) / SUM(h.total_households) AS prop_family_lone_parent,

    -- Other households
    SUM(h.other_family_household) / SUM(h.total_households) AS prop_other_family_household,
    SUM(h.other_household_types) / SUM(h.total_households) AS prop_other_household_types

FROM 
    oa_to_constituency_map o
INNER JOIN 
    household_composition_distribution_003 h
ON 
    o.OA21CD = h.geography_code
GROUP BY 
    o.PCON25CD
"""


CTE_QUERIES_LIST = [
    age_distribution_per_constituency_query,
    qualification_distribution_per_constituency_query,
    deprivation_distribution_per_constituency_query,
    economic_activity_distribution_per_constituency_query,
    ethnic_group_distribution_per_constituency_query,
    household_composition_distribution_per_constituency_query
]

FEATURES_2021 = [
    "prop_20_to_24", "prop_65_to_69", "prop_70_to_74", "prop_no_qualifications", 
    "prop_level_4_and_above", "prop_not_deprived", "prop_deprived_two_dimensions", 
    "prop_active_employment", "prop_unemployed", "prop_inactive", "prop_asian", 
    "prop_black", "prop_white", "prop_other", "prop_one_person", 
    "prop_family_married_couple", "prop_family_lone_parent"]


FEATURES_INTERPOLATED = [
    "prop_20_to_24", "prop_65_to_69", "prop_70_to_74", "frac_no_qualifications_2024", 
    "frac_level_4_and_above_2024", "frac_not_deprived_2024", "frac_deprived_2_dimensions_2024", 
    "frac_active_employment_2024", "frac_unemployed_2024", "frac_inactive_2024", "frac_asian_total_2024", 
    "frac_black_total_2024", "frac_white_british_2024", "frac_other_total_2024", "frac_one_person_2024",
    "frac_married_couple_2024", "frac_lone_parent_2024"]