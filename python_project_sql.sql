-- MySQL Workbench Forward Engineering

SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';

-- -----------------------------------------------------
-- Schema python_final_project
-- -----------------------------------------------------

-- -----------------------------------------------------
-- Schema python_final_project
-- -----------------------------------------------------
CREATE SCHEMA IF NOT EXISTS `python_final_project` DEFAULT CHARACTER SET utf8mb3 ;
USE `python_final_project` ;

-- -----------------------------------------------------
-- Table `python_final_project`.`users`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `python_final_project`.`users` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `first_name` VARCHAR(45) NULL DEFAULT NULL,
  `last_name` VARCHAR(45) NULL DEFAULT NULL,
  `email` VARCHAR(45) NULL DEFAULT NULL,
  `password` VARCHAR(60) NULL DEFAULT NULL,
  `created_at` DATETIME NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` DATETIME NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`))
ENGINE = InnoDB
AUTO_INCREMENT = 2
DEFAULT CHARACTER SET = utf8mb3;


-- -----------------------------------------------------
-- Table `python_final_project`.`user_data`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `python_final_project`.`user_data` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `date` DATE NULL DEFAULT NULL,
  `spcpi` FLOAT NULL DEFAULT NULL,
  `spcpi_m_s` FLOAT NULL DEFAULT NULL,
  `spcpi_m_fe` FLOAT NULL DEFAULT NULL,
  `spcpi_m_fes` FLOAT NULL DEFAULT NULL,
  `trim_mean_pce` FLOAT NULL DEFAULT NULL,
  `sixteenP_trim_mean_cpi` FLOAT NULL DEFAULT NULL,
  `median_cpi` FLOAT NULL DEFAULT NULL,
  `fpcpi` FLOAT NULL DEFAULT NULL,
  `fpcpi_m_fe` FLOAT NULL DEFAULT NULL,
  `created_at` DATETIME NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` DATETIME NULL DEFAULT CURRENT_TIMESTAMP,
  `user_id` INT NOT NULL,
  PRIMARY KEY (`id`),
  INDEX `fk_risk_analysis_users2_idx` (`user_id` ASC) VISIBLE,
  CONSTRAINT `fk_risk_analysis_users2`
    FOREIGN KEY (`user_id`)
    REFERENCES `python_final_project`.`users` (`id`))
ENGINE = InnoDB
AUTO_INCREMENT = 7
DEFAULT CHARACTER SET = utf8mb3;


-- -----------------------------------------------------
-- Table `python_final_project`.`results`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `python_final_project`.`results` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `spcpi` INT NULL DEFAULT NULL,
  `spcpi_m_s` INT NULL DEFAULT NULL,
  `spcpi_m_fe` INT NULL DEFAULT NULL,
  `spcpi_m_fes` INT NULL DEFAULT NULL,
  `trim_mean_pce` INT NULL DEFAULT NULL,
  `sixteenP_trim_mean_cpi` INT NULL DEFAULT NULL,
  `median_cpi` INT NULL DEFAULT NULL,
  `fpcpi` INT NULL DEFAULT NULL,
  `fpcpi_m_fe` INT NULL DEFAULT NULL,
  `mean` FLOAT NULL DEFAULT NULL,
  `created_at` DATETIME NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` DATETIME NULL DEFAULT CURRENT_TIMESTAMP,
  `user_id` INT NOT NULL,
  `user_data_id` INT NOT NULL,
  PRIMARY KEY (`id`),
  INDEX `fk_risk_analysis_users2_idx` (`user_id` ASC) VISIBLE,
  INDEX `fk_results_risk_analysis1_idx` (`user_data_id` ASC) VISIBLE,
  CONSTRAINT `fk_results_risk_analysis1`
    FOREIGN KEY (`user_data_id`)
    REFERENCES `python_final_project`.`user_data` (`id`),
  CONSTRAINT `fk_risk_analysis_users20`
    FOREIGN KEY (`user_id`)
    REFERENCES `python_final_project`.`users` (`id`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb3;


-- -----------------------------------------------------
-- Table `python_final_project`.`chatgpt_data`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `python_final_project`.`chatgpt_data` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `prompt` VARCHAR(500) NULL DEFAULT NULL,
  `completion` VARCHAR(500) NULL DEFAULT NULL,
  `created_at` DATETIME NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` DATETIME NULL DEFAULT CURRENT_TIMESTAMP,
  `user_id` INT NOT NULL,
  `result_id` INT NOT NULL,
  PRIMARY KEY (`id`),
  INDEX `fk_chatgpt_data_users1_idx` (`user_id` ASC) VISIBLE,
  INDEX `fk_chatgpt_data_results1_idx` (`result_id` ASC) VISIBLE,
  CONSTRAINT `fk_chatgpt_data_results1`
    FOREIGN KEY (`result_id`)
    REFERENCES `python_final_project`.`results` (`id`),
  CONSTRAINT `fk_chatgpt_data_users1`
    FOREIGN KEY (`user_id`)
    REFERENCES `python_final_project`.`users` (`id`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb3;


-- -----------------------------------------------------
-- Table `python_final_project`.`econ_data`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `python_final_project`.`econ_data` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `date` DATE NULL DEFAULT NULL,
  `spcpi` FLOAT NULL DEFAULT NULL,
  `spcpi_m_s` FLOAT NULL DEFAULT NULL,
  `spcpi_m_fe` FLOAT NULL DEFAULT NULL,
  `spcpi_m_fes` FLOAT NULL DEFAULT NULL,
  `trim_mean_pce` FLOAT NULL DEFAULT NULL,
  `sixteenP_trim_mean_cpi` FLOAT NULL DEFAULT NULL,
  `median_cpi` FLOAT NULL DEFAULT NULL,
  `fpcpi` FLOAT NULL DEFAULT NULL,
  `fpcpi_m_fe` FLOAT NULL DEFAULT NULL,
  `created_at` DATETIME NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` DATETIME NULL DEFAULT CURRENT_TIMESTAMP,
  `user_id` INT NOT NULL,
  PRIMARY KEY (`id`),
  INDEX `fk_data_users_idx` (`user_id` ASC) VISIBLE,
  CONSTRAINT `fk_data_users`
    FOREIGN KEY (`user_id`)
    REFERENCES `python_final_project`.`users` (`id`))
ENGINE = InnoDB
AUTO_INCREMENT = 471
DEFAULT CHARACTER SET = utf8mb3;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
